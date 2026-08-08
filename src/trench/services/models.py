from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

import requests

MODELS_DIR: Final[Path] = Path("models")

CONNECT_TIMEOUT_SECONDS: Final[float] = 10.0
READ_TIMEOUT_SECONDS: Final[float] = 30.0
DOWNLOAD_CHUNK_SIZE: Final[int] = 1024 * 1024
DEFAULT_MAX_RETRIES: Final[int] = 3

ProgressCallback = Callable[[int, int], None]


class ModelDownloadError(RuntimeError):
    """A model download failed, or the downloaded file failed checksum verification."""


class ModelDownloadCancelled(Exception):
    """A download was cancelled by the user before it completed."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str | None
    size_bytes: int
    notes: str

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.filename


REALESRGAN_X4PLUS: Final[ModelSpec] = ModelSpec(
    name="Real-ESRGAN x4plus",
    filename="RealESRGAN_x4plus.pth",
    url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    size_bytes=67_040_989,
    notes="4x upscaler applied to the extracted object. ~64 MB, fast on CPU or GPU.",
)

SAM_VIT_H: Final[ModelSpec] = ModelSpec(
    name="Segment Anything ViT-H",
    filename="sam_vit_h_4b8939.pth",
    url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    sha256="a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    size_bytes=2_564_550_879,
    notes=(
        "Highest-quality SAM checkpoint (~2.4 GB). GPU strongly recommended: mask "
        "generation over an entire image can take several minutes per image on CPU."
    ),
)

SAM_VIT_B: Final[ModelSpec] = ModelSpec(
    name="Segment Anything ViT-B",
    filename="sam_vit_b_01ec64.pth",
    url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    sha256="ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    size_bytes=375_042_383,
    notes=(
        "Smaller, faster SAM checkpoint (~358 MB). Coarser masks than ViT-H, but "
        "practical to run on a CPU-only machine. Used automatically without CUDA."
    ),
)

# Registry of selectable SAM checkpoints, largest/most-accurate first.
SAM_CHECKPOINTS: Final[dict[str, ModelSpec]] = {
    "vit_h": SAM_VIT_H,
    "vit_b": SAM_VIT_B,
}


def choose_sam_checkpoint(cuda_available: bool) -> tuple[str, ModelSpec]:
    """Pick a SAM checkpoint appropriate for the available hardware.

    Returns the (sam_model_registry key, ModelSpec) pair. ViT-H is the
    highest-quality checkpoint but is slow enough on CPU that a fresh user
    would assume the app had hung; ViT-B trades some mask quality for a
    runtime that stays reasonable without a GPU.
    """
    return ("vit_h", SAM_VIT_H) if cuda_available else ("vit_b", SAM_VIT_B)


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_model_valid(spec: ModelSpec) -> bool:
    """Return True if spec.path exists and matches its known checksum."""
    if not spec.path.exists():
        return False
    if spec.sha256 is None:
        return True
    return sha256_of(spec.path) == spec.sha256


def download_model(
    spec: ModelSpec,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> None:
    """Download `spec` to `spec.path`, verifying its checksum when known.

    Retries transient network failures up to `max_retries` times. Raises
    ModelDownloadCancelled if `cancel_event` is set mid-download, or
    ModelDownloadError if every attempt fails or the checksum never matches.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            _download_once(spec, progress_callback, cancel_event)
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            spec.path.unlink(missing_ok=True)
            continue

        if spec.sha256 is not None and sha256_of(spec.path) != spec.sha256:
            spec.path.unlink(missing_ok=True)
            last_error = ModelDownloadError(
                f"{spec.name}: downloaded file did not match the expected checksum"
            )
            continue

        return

    spec.path.unlink(missing_ok=True)
    raise ModelDownloadError(
        f"Failed to download {spec.name} after {max_retries} attempt(s): {last_error}"
    ) from last_error


def _download_once(
    spec: ModelSpec,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event | None,
) -> None:
    response = requests.get(
        spec.url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", spec.size_bytes))
    downloaded = 0

    # Download to a temp file first so a failed/cancelled attempt never
    # leaves a truncated file at spec.path for is_model_valid() to trust.
    tmp_path = spec.path.with_suffix(spec.path.suffix + ".part")
    try:
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if cancel_event is not None and cancel_event.is_set():
                    raise ModelDownloadCancelled(f"{spec.name} download cancelled")
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback is not None:
                    progress_callback(downloaded, total_size)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    tmp_path.replace(spec.path)
