from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIN_CROP_DIMENSION = 32


class CropValidationError(ValueError):
    """Raised when a requested manual crop cannot safely be assessed."""


@dataclass(frozen=True, slots=True)
class CropBox:
    left: int
    top: int
    width: int
    height: int


def validate_crop_box(
    image: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    minimum_dimension: int = MIN_CROP_DIMENSION,
) -> CropBox:
    """Validate a rectangular crop against an RGB image without changing that image."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise CropValidationError("Manual cropping requires an RGB image with three colour channels.")
    if minimum_dimension < 1:
        raise ValueError("Minimum crop dimension must be at least one pixel.")
    if min(left, top, width, height) < 0:
        raise CropValidationError("Crop coordinates must not be negative.")
    if width == 0 or height == 0:
        raise CropValidationError("Crop width and height must be greater than zero.")
    if width < minimum_dimension or height < minimum_dimension:
        raise CropValidationError(f"Crop is too small. Select an area at least {minimum_dimension} × {minimum_dimension} pixels.")

    image_height, image_width = image.shape[:2]
    if left + width > image_width or top + height > image_height:
        raise CropValidationError("Crop area must remain inside the original image boundaries.")
    return CropBox(left=left, top=top, width=width, height=height)


def crop_image(image: np.ndarray, box: CropBox, minimum_dimension: int = MIN_CROP_DIMENSION) -> np.ndarray:
    """Return an independent RGB crop after validating the supplied crop box."""
    valid_box = validate_crop_box(image, box.left, box.top, box.width, box.height, minimum_dimension)
    return image[valid_box.top : valid_box.top + valid_box.height, valid_box.left : valid_box.left + valid_box.width].copy()


def crop_from_component_box(image: np.ndarray, component_box: dict[str, float | int]) -> np.ndarray:
    """Convert streamlit-cropper's box format into a validated independent image crop."""
    try:
        return crop_image(
            image,
            CropBox(
                left=int(component_box["left"]),
                top=int(component_box["top"]),
                width=int(component_box["width"]),
                height=int(component_box["height"]),
            ),
        )
    except KeyError as exc:
        raise CropValidationError("Crop selection is incomplete. Drag the rectangle around the banana.") from exc