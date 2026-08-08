import cv2
import numpy as np
import torch

from trench.services import _torchvision_compat

_torchvision_compat.ensure_patched()

from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: E402
from cv2.typing import MatLike  # noqa: E402
from realesrgan import RealESRGANer  # noqa: E402


class ImageProcessor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.device: str = device
        self.upsampler: RealESRGANer = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=512,
            tile_pad=32,
            pre_pad=0,
            device=self.device,
            half=self.device == "cuda",
        )

    def upscale_image(self, img: MatLike) -> MatLike:
        # rgba => rgb for upscaling
        has_alpha: bool = img.shape[2] == 4
        if has_alpha:
            alpha = img[:, :, 3]
            img_rgb = img[:, :, :3]
        else:
            img_rgb = img
            alpha = None

        output, _ = self.upsampler.enhance(img_rgb, outscale=4)

        # restore alpha channel if it existed
        if alpha is not None:
            alpha_upscaled = cv2.resize(
                alpha,
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            output_rgba = np.dstack((output, alpha_upscaled))
            return output_rgba

        return output


def create_mask_preview(image: MatLike, mask: MatLike) -> MatLike:
    """Composite `image` (BGR) with the area outside `mask` greyed out, as RGBA."""
    highlighted = image.copy()
    overlay = np.zeros_like(image)
    overlay[mask == 0] = [128, 128, 128]

    alpha = 0.7
    cv2.addWeighted(highlighted, alpha, overlay, 1 - alpha, 0, highlighted)

    rgba = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGBA)
    alpha_channel = np.where(mask == 1, 255, 128).astype(np.uint8)
    rgba[:, :, 3] = alpha_channel

    return rgba


def apply_mask_alpha(image_bgr: MatLike, mask: MatLike) -> MatLike:
    """Return a BGRA copy of `image_bgr` with `mask` (0/255 per pixel) as the alpha channel."""
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    return rgba
