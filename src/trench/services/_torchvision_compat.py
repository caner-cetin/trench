"""Compatibility shim for basicsr 1.4.2 on torchvision >= 0.17.

basicsr's top-level package eagerly imports its `data` subpackage, which
imports `torchvision.transforms.functional_tensor` for `rgb_to_grayscale` —
a private module torchvision removed in 0.17 after moving that function to
the public `torchvision.transforms.functional` module. basicsr has not
published a fix (1.4.2 on PyPI predates the removal), and this project only
uses `basicsr.archs.rrdbnet_arch`, never the `data` subpackage that trips
over it. Call `ensure_patched()` before importing `basicsr` to register a
stand-in module so the eager import succeeds.
"""

from __future__ import annotations

import sys
import types


def ensure_patched() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    from torchvision.transforms import functional as _functional

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = _functional.rgb_to_grayscale  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules["torchvision.transforms.functional_tensor"] = shim
