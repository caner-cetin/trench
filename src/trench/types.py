from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cv2.typing import MatLike


@dataclass
class MaskRecord:
    """A single object mask produced by SAM's automatic mask generator.

    segmentation: binary HxW mask array.
    bbox: box around the mask, in XYWH format.
    area: area in pixels of the mask.
    predicted_iou: the model's own prediction of the mask's quality.
    point_coords: point coordinates input to the model to generate this mask.
    stability_score: a measure of the mask's quality under threshold perturbation.
    crop_box: the crop of the source image used to generate the mask, in XYWH format.
    """

    segmentation: MatLike
    bbox: list[float]
    area: int
    predicted_iou: float
    point_coords: list[list[float]]
    stability_score: float
    crop_box: list[float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskRecord:
        return cls(
            segmentation=data["segmentation"],
            bbox=data["bbox"],
            area=data["area"],
            predicted_iou=data["predicted_iou"],
            point_coords=data["point_coords"],
            stability_score=data["stability_score"],
            crop_box=data["crop_box"],
        )
