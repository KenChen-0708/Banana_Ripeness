from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycoprops
from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_STATE
from src.preprocessing import banana_mask, resize_rgb

from .base import BaseBananaPipeline


class GLCMMorphologyRandomForestPipeline(BaseBananaPipeline):
    key = "glcm_rf"
    display_name = "Morphology + GLCM + Random Forest"
    feature_version = "glcm-morphology-masked-v2"
    image_size = (128, 128)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = banana_mask(image, self.image_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

        quantised = (gray // 16).astype(np.uint8)
        matrix = _masked_graycomatrix(
            quantised, refined > 0, distances=(1, 2, 4),
            angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), levels=16,
        )
        texture: list[float] = []
        for prop in ("contrast", "correlation", "energy", "homogeneity"):
            values = graycoprops(matrix, prop)
            texture.extend([float(np.nanmean(values)), float(np.nanstd(values))])

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max(contours, key=cv2.contourArea) if contours else None
        area = float(cv2.contourArea(largest)) if largest is not None else 0.0
        perimeter = float(cv2.arcLength(largest, True)) if largest is not None else 0.0
        if largest is not None:
            _, _, width, height = cv2.boundingRect(largest)
        else:
            width = height = 1
        fruit_pixels = gray[refined > 0]
        if not fruit_pixels.size:
            fruit_pixels = gray.ravel()
        morphology = [
            area / refined.size,
            perimeter / (2.0 * sum(refined.shape)),
            width / max(1.0, float(height)),
            float(np.count_nonzero(refined)) / refined.size,
            float(fruit_pixels.mean()) / 255.0,
            float(fruit_pixels.std()) / 255.0,
        ]
        return np.asarray([*texture, *morphology], dtype=np.float32)

    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )


def _masked_graycomatrix(
    image: np.ndarray,
    mask: np.ndarray,
    distances: tuple[int, ...],
    angles: tuple[float, ...],
    levels: int,
) -> np.ndarray:
    """Build symmetric normalised GLCMs using only fruit-to-fruit pixel pairs."""
    height, width = image.shape
    result = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float64)
    for distance_index, distance in enumerate(distances):
        for angle_index, angle in enumerate(angles):
            dx = int(round(np.cos(angle) * distance))
            dy = int(round(np.sin(angle) * distance))
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            source = image[y0:y1, x0:x1]
            target = image[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            valid = mask[y0:y1, x0:x1] & mask[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            if np.any(valid):
                indices = source[valid].astype(np.int32) * levels + target[valid].astype(np.int32)
                counts = np.bincount(indices, minlength=levels * levels).reshape(levels, levels)
                counts = counts + counts.T
                result[:, :, distance_index, angle_index] = counts / counts.sum()
            else:
                result[0, 0, distance_index, angle_index] = 1.0
    return result

