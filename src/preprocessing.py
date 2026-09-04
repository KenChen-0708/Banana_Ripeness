from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def decode_image(source: str | Path | bytes | bytearray | BinaryIO) -> np.ndarray:
    """Decode an image as an EXIF-corrected RGB uint8 array."""
    try:
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        elif isinstance(source, (bytes, bytearray)):
            image = Image.open(BytesIO(source))
        else:
            image = Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width < 16 or image.height < 16:
            raise ImageValidationError("Image dimensions must be at least 16 x 16 pixels.")
        if image.width * image.height > 40_000_000:
            raise ImageValidationError("Image is too large; the maximum is 40 megapixels.")
        return np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ImageValidationError):
            raise
        raise ImageValidationError("The file is not a valid JPG or PNG image.") from exc


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    _validate_rgb(image)
    width, height = size
    interpolation = cv2.INTER_AREA if image.shape[1] > width or image.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def denoise_and_enhance(image: np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    """Apply median denoising and luminance-only CLAHE contrast enhancement.

    Enhancing only the L* channel avoids inventing colour shifts that could alter
    the visual ripeness cues. The operation is deterministic and keeps uint8 RGB
    output for downstream OpenCV routines.
    """
    _validate_rgb(image)
    rgb = resize_rgb(image, size) if size is not None else image.copy()
    denoised = cv2.medianBlur(rgb, 5)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced_lightness, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def _validate_rgb(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("Expected an RGB image with three colour channels.")
    if image.dtype != np.uint8:
        raise ImageValidationError("Expected an 8-bit RGB image.")


def banana_mask(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Estimate the largest fruit-like foreground region reproducibly.

    The border colour models the background. Saturated green/yellow/brown pixels
    supplement the colour-distance mask, followed by morphological clean-up.
    """
    rgb = resize_rgb(image, size)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    border = max(2, min(rgb.shape[:2]) // 16)
    border_pixels = np.concatenate(
        [lab[:border].reshape(-1, 3), lab[-border:].reshape(-1, 3), lab[:, :border].reshape(-1, 3), lab[:, -border:].reshape(-1, 3)]
    )
    background = np.median(border_pixels, axis=0)
    colour_distance = np.linalg.norm(lab - background, axis=2)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    chromatic_fruit = (saturation >= 35) & (value >= 25) & (hue <= 95)
    candidate = ((colour_distance >= 18) | chromatic_fruit) & (value >= 18)
    mask = (candidate.astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    area_ratio = float(np.count_nonzero(mask)) / mask.size
    if area_ratio < 0.015:
        return np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def colour_masks(image: np.ndarray, mask: np.ndarray | None = None, size: tuple[int, int] = (224, 224)) -> dict[str, np.ndarray]:
    rgb = resize_rgb(image, size)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if mask is None:
        mask = banana_mask(image, size)
    fruit = mask > 0
    h, s, v = cv2.split(hsv)

    green = fruit & (h >= 32) & (h <= 95) & (s >= 45) & (v >= 35)
    yellow = fruit & (h >= 18) & (h < 40) & (s >= 45) & (v >= 60)
    brown = fruit & (h >= 3) & (h <= 24) & (s >= 35) & (v >= 20) & (v <= 190)
    dark = fruit & (v < 65)
    return {
        "fruit": mask,
        "green": green.astype(np.uint8) * 255,
        "yellow": yellow.astype(np.uint8) * 255,
        "brown": brown.astype(np.uint8) * 255,
        "dark": dark.astype(np.uint8) * 255,
    }


def brown_spot_ratio(image: np.ndarray, mask: np.ndarray | None = None, size: tuple[int, int] = (224, 224)) -> float:
    masks = colour_masks(image, mask=mask, size=size)
    fruit_pixels = max(1, int(np.count_nonzero(masks["fruit"])))
    blemish = cv2.bitwise_or(masks["brown"], masks["dark"])
    return float(np.count_nonzero(blemish)) / fruit_pixels


def mask_overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (255, 87, 51)) -> np.ndarray:
    rgb = resize_rgb(image, (mask.shape[1], mask.shape[0]))
    overlay = rgb.copy()
    overlay[mask > 0] = colour
    return cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)


def rectify_object(image: np.ndarray, mask: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Crop the detected fruit and letterbox it without aspect-ratio distortion."""
    rgb = resize_rgb(image, (mask.shape[1], mask.shape[0]))
    points = cv2.findNonZero(mask)
    if points is None:
        return resize_rgb(rgb, size)
    x, y, width, height = cv2.boundingRect(points)
    crop = rgb[y : y + height, x : x + width]
    target_width, target_height = size
    scale = min(target_width / max(1, width), target_height / max(1, height))
    fitted_width = max(1, int(round(width * scale)))
    fitted_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    fitted = cv2.resize(crop, (fitted_width, fitted_height), interpolation=interpolation)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    left = (target_width - fitted_width) // 2
    top = (target_height - fitted_height) // 2
    canvas[top : top + fitted_height, left : left + fitted_width] = fitted
    return canvas


def analyse_object(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> dict[str, object]:
    """Return auditable preprocessing, detection, and surface measurements."""
    standardised = resize_rgb(image, size)
    enhanced = denoise_and_enhance(image, size)
    mask = banana_mask(enhanced, size)
    masks = colour_masks(enhanced, mask=mask, size=size)
    blemish = cv2.bitwise_or(masks["brown"], masks["dark"])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None

    detection = standardised.copy()
    if largest is None:
        x = y = 0
        width, height = size
        contour_area = perimeter = 0.0
    else:
        x, y, width, height = cv2.boundingRect(largest)
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        cv2.drawContours(detection, [largest], -1, (55, 210, 110), 2)
        cv2.rectangle(detection, (x, y), (x + width - 1, y + height - 1), (255, 190, 35), 2)

    fruit_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    return {
        "standardised_rgb": standardised,
        "enhanced_rgb": enhanced,
        "rectified_rgb": rectify_object(enhanced, mask, size),
        "fruit_mask": mask,
        "brown_mask": blemish,
        "detection_overlay": detection,
        "bounding_box": (int(x), int(y), int(width), int(height)),
        "contour_area_px": contour_area,
        "perimeter_px": perimeter,
        "fruit_area_px": fruit_pixels,
        "fruit_area_ratio": fruit_pixels / max(1, total_pixels),
        "blemish_ratio": float(np.count_nonzero(blemish)) / max(1, fruit_pixels),
        "green_ratio": float(np.count_nonzero(masks["green"])) / max(1, fruit_pixels),
        "yellow_ratio": float(np.count_nonzero(masks["yellow"])) / max(1, fruit_pixels),
        "major_dimension_px": float(max(width, height)),
        "minor_dimension_px": float(min(width, height)),
    }

