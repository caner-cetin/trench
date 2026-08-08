"""Utilities for preparing transparent-object export destinations."""

from os import PathLike
from pathlib import Path


class ExportPathError(RuntimeError):
    """Raised when an export directory cannot be prepared."""


def prepare_export_path(output_dir: str | PathLike[str], object_number: int) -> Path:
    """Create ``output_dir`` and return an unused PNG path for an object.

    Existing exports are preserved by adding a numeric suffix, for example
    ``object_2_1.png`` after ``object_2.png`` already exists.
    """
    directory = Path(output_dir).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ExportPathError(
            f"Could not create export folder '{directory}': {error}"
        ) from error

    stem = f"object_{object_number}"
    output_path = directory / f"{stem}.png"
    suffix = 1
    while output_path.exists():
        output_path = directory / f"{stem}_{suffix}.png"
        suffix += 1

    return output_path
