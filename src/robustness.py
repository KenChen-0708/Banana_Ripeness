from __future__ import annotations

import cv2
import numpy as np


def apply_perturbation(image: np.ndarray, name: str, random_state: int = 42) -> np.ndarray:
    """Apply a deterministic acquisition perturbation for robustness testing."""
    if name == "clean":
        return image.copy()
    if name == "dim_30_percent":
        return cv2.convertScaleAbs(image, alpha=0.70, beta=0)
    if name == "bright_30_percent":
        return cv2.convertScaleAbs(image, alpha=1.30, beta=0)
    if name == "gaussian_blur":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if name == "sensor_noise":
        generator = np.random.default_rng(random_state)
        noise = generator.normal(0, 12, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "rotation_10_degrees":
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 10.0, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    raise ValueError(f"Unknown perturbation: {name}")


PERTURBATIONS = ("clean", "dim_30_percent", "bright_30_percent", "gaussian_blur", "sensor_noise", "rotation_10_degrees")
