from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw


MIN_CROP_DIMENSION = 32
MAX_CROP_EDITOR_DIMENSION = 420
GALLERY_THUMBNAIL_SIZE = (220, 150)


class CropValidationError(ValueError):
    """Raised when a requested manual crop cannot safely be assessed."""


@dataclass(frozen=True, slots=True)
class CropBox:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CropPolygon:
    points: tuple[tuple[int, int], ...]


def _segments_intersect(
    first_start: tuple[int, int],
    first_end: tuple[int, int],
    second_start: tuple[int, int],
    second_end: tuple[int, int],
) -> bool:
    def orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
        value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        return 0 if value == 0 else 1 if value > 0 else 2

    def on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(first_start, second_start, first_end))
        or (o2 == 0 and on_segment(first_start, second_end, first_end))
        or (o3 == 0 and on_segment(second_start, first_start, second_end))
        or (o4 == 0 and on_segment(second_start, first_end, second_end))
    )


def validate_crop_polygon(
    image: np.ndarray,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    minimum_dimension: int = MIN_CROP_DIMENSION,
) -> CropPolygon:
    """Validate a polygon selection against an RGB image."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise CropValidationError("Polygon cropping requires an RGB image with three colour channels.")
    if len(points) < 3:
        raise CropValidationError("Place at least three points around the banana.")

    height, width = image.shape[:2]
    normalized = tuple((round(float(x)), round(float(y))) for x, y in points)
    if len(set(normalized)) < 3:
        raise CropValidationError("The polygon needs at least three different points.")
    if any(x < 0 or y < 0 or x >= width or y >= height for x, y in normalized):
        raise CropValidationError("Polygon points must remain inside the image boundaries.")

    edge_count = len(normalized)
    for first_index in range(edge_count):
        first_edge = (normalized[first_index], normalized[(first_index + 1) % edge_count])
        for second_index in range(first_index + 1, edge_count):
            if second_index in {first_index, first_index + 1} or (first_index == 0 and second_index == edge_count - 1):
                continue
            second_edge = (normalized[second_index], normalized[(second_index + 1) % edge_count])
            if _segments_intersect(*first_edge, *second_edge):
                raise CropValidationError("Polygon edges must not cross each other.")

    xs, ys = zip(*normalized, strict=True)
    if max(xs) - min(xs) + 1 < minimum_dimension or max(ys) - min(ys) + 1 < minimum_dimension:
        raise CropValidationError(
            f"Polygon is too small. Select an area at least {minimum_dimension} × {minimum_dimension} pixels."
        )
    area = abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(normalized, normalized[1:] + normalized[:1], strict=True)
        )
    ) / 2.0
    if area < minimum_dimension * minimum_dimension / 2:
        raise CropValidationError("Polygon area is too small.")
    return CropPolygon(points=normalized)


def crop_polygon_image(
    image: np.ndarray,
    polygon: CropPolygon,
    background: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Mask pixels outside a validated polygon and crop to its bounding rectangle."""
    valid_polygon = validate_crop_polygon(image, polygon.points)
    mask_image = Image.new("L", (image.shape[1], image.shape[0]), 0)
    ImageDraw.Draw(mask_image).polygon(valid_polygon.points, fill=255)
    mask = np.asarray(mask_image) > 0
    masked = np.empty_like(image)
    masked[:] = background
    masked[mask] = image[mask]
    xs, ys = zip(*valid_polygon.points, strict=True)
    left, top = min(xs), min(ys)
    right, bottom = max(xs) + 1, max(ys) + 1
    return masked[top:bottom, left:right].copy()


def polygon_to_display(
    polygon: CropPolygon, original_image: np.ndarray, display_image: np.ndarray
) -> list[tuple[float, float]]:
    """Map a full-resolution polygon to editor display coordinates."""
    original_height, original_width = original_image.shape[:2]
    display_height, display_width = display_image.shape[:2]
    return [
        (x * display_width / original_width, y * display_height / original_height)
        for x, y in polygon.points
    ]


def polygon_from_display(
    points: list[list[float]] | list[tuple[float, float]],
    original_image: np.ndarray,
    display_image: np.ndarray,
) -> CropPolygon:
    """Map editor polygon points back to the original image resolution."""
    original_height, original_width = original_image.shape[:2]
    display_height, display_width = display_image.shape[:2]
    mapped = [
        (
            min(original_width - 1, max(0, x * original_width / display_width)),
            min(original_height - 1, max(0, y * original_height / display_height)),
        )
        for x, y in points
    ]
    return validate_crop_polygon(original_image, mapped)


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


def gallery_thumbnail(image: np.ndarray, size: tuple[int, int] = GALLERY_THUMBNAIL_SIZE) -> np.ndarray:
    """Create a fixed-size, aspect-ratio-preserving display thumbnail for the upload gallery."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise CropValidationError("Gallery thumbnails require an RGB image with three colour channels.")
    width, height = size
    if width < 1 or height < 1:
        raise ValueError("Gallery thumbnail dimensions must be positive.")
    source = Image.fromarray(image)
    scale = min(width / source.width, height / source.height)
    scaled_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    resized = source.resize(scaled_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, color=(10, 25, 27))
    offset = ((width - scaled_size[0]) // 2, (height - scaled_size[1]) // 2)
    canvas.paste(resized, offset)
    return np.asarray(canvas)


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


def confirm_polygon_crop_state(
    states: Mapping[str, dict], filename: str, image: np.ndarray, polygon: CropPolygon
) -> dict[str, dict]:
    """Store a full-resolution polygon crop for one image."""
    key = image_crop_key(filename, image)
    valid_polygon = validate_crop_polygon(image, polygon.points)
    updated = dict(states)
    updated[key] = {
        "source_identity": key,
        "filename": filename,
        "crop_polygon": valid_polygon,
        "confirmed_crop": crop_polygon_image(image, valid_polygon),
        "status": "polygon_cropped",
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
        analysis_images[filename] = (
            state["confirmed_crop"]
            if state and state.get("status") in {"cropped", "polygon_cropped"}
            else image
        )
    return analysis_images
