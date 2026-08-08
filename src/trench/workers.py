import threading
from typing import Any, final, override

import numpy as np
from cv2.typing import MatLike
from PySide6.QtCore import QThread, Signal
from segment_anything import SamAutomaticMaskGenerator

from trench.services.image_processor import ImageProcessor
from trench.services.models import (
    ModelDownloadCancelled,
    ModelDownloadError,
    ModelSpec,
    download_model,
)
from trench.types import MaskRecord


@final
class MaskWorker(QThread):
    finished_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(
        self, image: MatLike, mask_generator: SamAutomaticMaskGenerator
    ) -> None:
        super().__init__()
        self.image: MatLike = image
        self.mask_generator: SamAutomaticMaskGenerator = mask_generator

    @override
    def run(self) -> None:
        try:
            masks: list[dict[str, Any]] = self.mask_generator.generate(self.image)
            significant_masks: list[MaskRecord] = [
                MaskRecord.from_dict(mask) for mask in masks if mask["area"] > 1000
            ]
        except Exception as error:
            message = str(error).strip() or type(error).__name__
            self.error_signal.emit(f"Mask generation failed: {message}")
            return

        self.finished_signal.emit(significant_masks)


@final
class UpscaleWorker(QThread):
    finished_signal = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self, processor: ImageProcessor, rgba: MatLike) -> None:
        super().__init__()
        self.processor: ImageProcessor = processor
        self.rgba = rgba

    @override
    def run(self) -> None:
        try:
            upscaled_rgba = self.processor.upscale_image(self.rgba)
        except Exception as error:
            message = str(error).strip() or type(error).__name__
            self.error_signal.emit(f"Object upscaling failed: {message}")
            return

        self.finished_signal.emit(upscaled_rgba)


@final
class ModelDownloadWorker(QThread):
    """Downloads and verifies a single model checkpoint off the UI thread."""

    progress_signal = Signal(int, int)  # downloaded_bytes, total_bytes
    error_signal = Signal(str)
    finished_signal = Signal(bool)  # True on success, False on cancel/error

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec: ModelSpec = spec
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    @override
    def run(self) -> None:
        try:
            download_model(
                self.spec,
                progress_callback=self.progress_signal.emit,
                cancel_event=self._cancel_event,
            )
        except ModelDownloadCancelled:
            self.finished_signal.emit(False)
            return
        except ModelDownloadError as exc:
            self.error_signal.emit(str(exc))
            self.finished_signal.emit(False)
            return
        self.finished_signal.emit(True)
