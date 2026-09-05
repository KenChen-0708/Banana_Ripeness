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


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian noise reduction with odd kernel size."""
    _validate_rgb(image)
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to remove impulse/salt-and-pepper noise."""
    _validate_rgb(image)
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(image, k)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Apply edge-preserving bilateral smoothing filter."""
    _validate_rgb(image)
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_contrast_stretching(image: np.ndarray) -> np.ndarray:
    """Linear min-max normalization/contrast stretching per channel."""
    _validate_rgb(image)
    stretched = np.zeros_like(image, dtype=np.uint8)
    for channel in range(3):
        c_min, c_max = float(image[:, :, channel].min()), float(image[:, :, channel].max())
        if c_max > c_min:
            stretched[:, :, channel] = np.clip(
                (image[:, :, channel].astype(np.float32) - c_min) * (255.0 / (c_max - c_min)),
                0,
                255,
            ).astype(np.uint8)
        else:
            stretched[:, :, channel] = image[:, :, channel]
    return stretched


def apply_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization on the LAB L-channel."""
    _validate_rgb(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def apply_morphology(
    mask: np.ndarray,
    op: str = "close",
    shape: str = "ellipse",
    kernel_size: int = 5,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operation (open, close, dilate, erode) to a binary mask."""
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    struct_shape = cv2.MORPH_ELLIPSE if shape == "ellipse" else cv2.MORPH_RECT if shape == "rect" else cv2.MORPH_CROSS
    kernel = cv2.getStructuringElement(struct_shape, (k, k))
    operations = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "dilate": cv2.MORPH_DILATE,
        "erode": cv2.MORPH_ERODE,
    }
    morph_op = operations.get(op.lower(), cv2.MORPH_CLOSE)
    return cv2.morphologyEx(mask, morph_op, kernel, iterations=iterations)


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

    b_channel = lab[:, :, 2]
    chromatic_fruit = (saturation >= 30) & (value >= 25) & (hue <= 95)
    
    # Adaptive background & shadow suppression
    if background[2] < 126:  # Cool / blue / lilac background (b* < 126)
        # Fruit must have warm/yellow/green tone (b* >= 132) to reject cool cast shadows
        candidate = (b_channel >= 132) & (colour_distance >= 18) & (value >= 25)
    elif background[0] < 120:  # Dark / black surface (L* < 120)
        is_shadow = (value < background[0] + 15) & (saturation < 30)
        candidate = ((colour_distance >= 22) & ~is_shadow) | chromatic_fruit
        candidate = candidate & (value >= 25) & ~is_shadow
    else:  # Standard light / white studio background
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


def colour_masks(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    size: tuple[int, int] = (224, 224),
    brown_h_range: tuple[int, int] = (3, 24),
    dark_v_thresh: int = 65,
) -> dict[str, np.ndarray]:
    rgb = resize_rgb(image, size)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if mask is None:
        mask = banana_mask(image, size)
    elif mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)
    fruit = mask > 0
    h, s, v = cv2.split(hsv)

    green = fruit & (h >= 28) & (h <= 95) & (s >= 35) & (v >= 25)
    yellow = fruit & (h >= 16) & (h < 38) & (s >= 40) & (v >= 50)
    brown = fruit & (h >= brown_h_range[0]) & (h <= brown_h_range[1]) & (s >= 35) & (v >= 20) & (v <= 190)
    
    # Exclude healthy green chlorophyll from false blemish detection
    is_chlorophyll = (h >= 28) & (h <= 95) & (s >= 25)
    dark = fruit & (v < dark_v_thresh) & (s >= 15) & ~is_chlorophyll & ((h <= 26) | (v < 38))
    return {
        "fruit": mask,
        "green": green.astype(np.uint8) * 255,
        "yellow": yellow.astype(np.uint8) * 255,
        "brown": brown.astype(np.uint8) * 255,
        "dark": dark.astype(np.uint8) * 255,
    }


def brown_spot_ratio(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    size: tuple[int, int] = (224, 224),
    brown_h_range: tuple[int, int] = (3, 24),
    dark_v_thresh: int = 65,
) -> float:
    masks = colour_masks(image, mask=mask, size=size, brown_h_range=brown_h_range, dark_v_thresh=dark_v_thresh)
    fruit_pixels = max(1, int(np.count_nonzero(masks["fruit"])))
    blemish = cv2.bitwise_or(masks["brown"], masks["dark"])
    return float(np.count_nonzero(blemish)) / fruit_pixels


def determine_quality_grade(predicted_class: str, blemish_ratio: float) -> str:
    """Classify commercial quality grading based on defect ratio and predicted class."""
    if predicted_class.lower() == "rotten" or blemish_ratio >= 0.30:
        return "Reject / Rotten"
    if blemish_ratio < 0.05:
        return "Grade A (Premium)"
    if blemish_ratio < 0.15:
        return "Grade B (Standard)"
    return "Grade C (Processing)"


def calibrate_fruit_geometry(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    pixels_per_cm: float = 10.0,
    size: tuple[int, int] = (224, 224),
) -> dict:
    """Perform spatial calibration, contour detection, and metric estimation in cm / cm²."""
    rgb = resize_rgb(image, size)
    if mask is None:
        mask = banana_mask(rgb, size)
    elif mask.shape[:2] != (size[1], size[0]):
        mask = cv2.resize(mask, (size[0], size[1]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    pixel_area = float(cv2.contourArea(largest_contour)) if largest_contour is not None else float(np.count_nonzero(mask))
    pixel_perimeter = float(cv2.arcLength(largest_contour, True)) if largest_contour is not None else 0.0

    px_cm = max(0.1, float(pixels_per_cm))
    cm2_per_px2 = 1.0 / (px_cm * px_cm)
    cm_per_px = 1.0 / px_cm

    if largest_contour is not None and len(largest_contour) >= 5:
        bx, by, bw, bh = cv2.boundingRect(largest_contour)
        moments = cv2.moments(largest_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = bx + bw // 2, by + bh // 2

        # Rotated rect for true length along fruit orientation
        rect = cv2.minAreaRect(largest_contour)
        (rcx, rcy), (rw, rh), angle = rect
        major_len_px = max(rw, rh)
        minor_width_px = min(rw, rh)
    else:
        bx, by, bw, bh = 0, 0, size[0], size[1]
        cx, cy = size[0] // 2, size[1] // 2
        major_len_px = max(bw, bh)
        minor_width_px = min(bw, bh)
        rect = ((cx, cy), (bw, bh), 0)

    phys_length_cm = major_len_px * cm_per_px
    phys_width_cm = minor_width_px * cm_per_px
    phys_area_cm2 = pixel_area * cm2_per_px2
    phys_perimeter_cm = pixel_perimeter * cm_per_px
    aspect_ratio = major_len_px / max(1.0, minor_width_px)

    blem_ratio = brown_spot_ratio(rgb, mask=mask, size=size)
    blem_area_cm2 = phys_area_cm2 * blem_ratio

    return {
        "pixels_per_cm": px_cm,
        "pixel_area": pixel_area,
        "physical_area_cm2": phys_area_cm2,
        "pixel_perimeter": pixel_perimeter,
        "physical_perimeter_cm": phys_perimeter_cm,
        "bbox_xywh": (bx, by, bw, bh),
        "min_area_rect": rect,
        "physical_length_cm": phys_length_cm,
        "physical_width_cm": phys_width_cm,
        "aspect_ratio": aspect_ratio,
        "centroid": (cx, cy),
        "blemish_ratio": blem_ratio,
        "blemish_area_cm2": blem_area_cm2,
        "quality_grade": determine_quality_grade("ripe", blem_ratio),
        "contour": largest_contour,
    }


def draw_annotated_geometry(
    image: np.ndarray,
    geometry: dict,
    size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Draw calibrated bounding box, rotated rect, contour, and dimension metrics onto RGB image."""
    annotated = resize_rgb(image, size).copy()
    contour = geometry.get("contour")
    if contour is not None:
        cv2.drawContours(annotated, [contour], -1, (46, 204, 113), 2)  # Emerald green contour

    # Draw Minimum Area Bounding Box
    rect = geometry.get("min_area_rect")
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(annotated, [box], 0, (241, 196, 15), 2)  # Yellow calibrated bbox

    # Draw Centroid
    cx, cy = geometry.get("centroid", (size[0] // 2, size[1] // 2))
    cv2.circle(annotated, (cx, cy), 4, (231, 76, 60), -1)  # Red center
    cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)

    # Annotate dimensions text
    length_cm = geometry.get("physical_length_cm", 0.0)
    width_cm = geometry.get("physical_width_cm", 0.0)
    area_cm2 = geometry.get("physical_area_cm2", 0.0)
    label = f"{length_cm:.1f}cm x {width_cm:.1f}cm ({area_cm2:.1f}cm2)"

    bx, by, _, _ = geometry.get("bbox_xywh", (10, 10, 50, 50))
    text_y = max(18, by - 6)
    cv2.putText(
        annotated,
        label,
        (max(5, bx), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        label,
        (max(5, bx), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (24, 74, 53),
        1,
        cv2.LINE_AA,
    )
    return annotated


def mask_overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (255, 87, 51)) -> np.ndarray:
    rgb = resize_rgb(image, (mask.shape[1], mask.shape[0]))
    overlay = rgb.copy()
    overlay[mask > 0] = colour
    return cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)

