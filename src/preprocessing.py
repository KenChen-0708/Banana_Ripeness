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

