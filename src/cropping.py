from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import numpy as np
from PIL import Image


MIN_CROP_DIMENSION = 32
MAX_CROP_EDITOR_DIMENSION = 420


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


def resize_for_crop_editor(image: np.ndarray, maximum_dimension: int = MAX_CROP_EDITOR_DIMENSION) -> np.ndarray:
    """Create a bounded display-only editor image without changing the analysis input."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise CropValidationError("Manual cropping requires an RGB image with three colour channels.")
    if maximum_dimension < 1:
        raise ValueError("Maximum editor dimension must be at least one pixel.")
    height, width = image.shape[:2]
    scale = min(1.0, maximum_dimension / max(width, height))
    display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if display_size == (width, height):
        return image.copy()
    return np.asarray(Image.fromarray(image).resize(display_size, Image.Resampling.LANCZOS))


def crop_box_to_display(box: CropBox, original_image: np.ndarray, display_image: np.ndarray) -> tuple[int, int, int, int]:
    """Map an original-image crop rectangle to cropper's display coordinates."""
    original_height, original_width = original_image.shape[:2]
    display_height, display_width = display_image.shape[:2]
    scale_x, scale_y = display_width / original_width, display_height / original_height
    left, top = round(box.left * scale_x), round(box.top * scale_y)
    right, bottom = round((box.left + box.width) * scale_x), round((box.top + box.height) * scale_y)
    return left, right, top, bottom


def crop_box_from_display(component_box: dict[str, float | int], original_image: np.ndarray, display_image: np.ndarray) -> CropBox:
    """Map cropper display coordinates back to a validated full-resolution crop rectangle."""
    try:
        display_left = float(component_box["left"])
        display_top = float(component_box["top"])
        display_width = float(component_box["width"])
        component_height = float(component_box["height"])
    except KeyError as exc:
        raise CropValidationError("Crop selection is incomplete. Drag the rectangle around the banana.") from exc

    original_height, original_width = original_image.shape[:2]
    editor_height, editor_width = display_image.shape[:2]
    scale_x, scale_y = original_width / editor_width, original_height / editor_height
    left, top = round(display_left * scale_x), round(display_top * scale_y)
    right = round((display_left + display_width) * scale_x)
    bottom = round((display_top + component_height) * scale_y)
    left, top = max(0, left), max(0, top)
    right, bottom = min(original_width, right), min(original_height, bottom)
    return validate_crop_box(original_image, left, top, right - left, bottom - top)


def image_crop_key(filename: str, image: np.ndarray) -> str:
    """Create a stable per-upload identity from its filename and source pixels."""
    if not isinstance(image, np.ndarray):
        raise CropValidationError("A NumPy image is required to create a crop identity.")
    digest = sha256(image.tobytes()).hexdigest()
    return f"{filename}:{image.shape}:{digest}"


def reconcile_crop_states(states: Mapping[str, dict], images: Mapping[str, np.ndarray]) -> dict[str, dict]:
    """Keep only crop states whose original upload remains present and unchanged."""
    active_keys = {image_crop_key(filename, image) for filename, image in images.items()}
    return {key: value for key, value in states.items() if key in active_keys}


def confirm_crop_state(states: Mapping[str, dict], filename: str, image: np.ndarray, box: CropBox) -> dict[str, dict]:
    """Store an independent full-resolution crop and its rectangle for one image."""
    key = image_crop_key(filename, image)
    valid_box = validate_crop_box(image, box.left, box.top, box.width, box.height)
    updated = dict(states)
    updated[key] = {
        "source_identity": key,
        "filename": filename,
        "crop_box": valid_box,
        "confirmed_crop": crop_image(image, valid_box),
        "status": "cropped",
    }
    return updated


def reset_crop_state(states: Mapping[str, dict], filename: str, image: np.ndarray) -> dict[str, dict]:
    """Remove only one image's crop state."""
    updated = dict(states)
    updated.pop(image_crop_key(filename, image), None)
    return updated


def analysis_images_for_assessment(images: Mapping[str, np.ndarray], states: Mapping[str, dict]) -> dict[str, np.ndarray]:
    """Choose each confirmed full-resolution crop, otherwise retain the original input image."""
    analysis_images = {}
    for filename, image in images.items():
        state = states.get(image_crop_key(filename, image))
        analysis_images[filename] = state["confirmed_crop"] if state and state.get("status") == "cropped" else image
    return analysis_images