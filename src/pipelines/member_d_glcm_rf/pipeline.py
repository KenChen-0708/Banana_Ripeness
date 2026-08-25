"""Member D Task: GLCM Texture Feature Extraction + Morphological Processing + Random Forest.

Student: Lee Siu Yee
Report Section: 2.3 Gray-Level Co-occurrence Matrix (GLCM) + Morphological Processing + Random Forest Classifier
Algorithm Overview:
    - Preprocessing: Grayscale conversion, morphological closing and opening for mask cleanup.
    - Texture Analysis: Gray-Level Co-occurrence Matrix (GLCM) across 4 angles (0°, 45°, 90°, 135°) and 3 distances (1, 2, 4).
    - Features: Contrast, Correlation, Energy, Homogeneity (mean and std) combined with morphological area, perimeter, and aspect ratio.
    - Classification: 250-tree Random Forest Classifier with balanced class weights.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_STATE
from src.pipelines.base import BaseBananaPipeline
from src.preprocessing import banana_mask, resize_rgb


class GLCMMorphologyRandomForestPipeline(BaseBananaPipeline):
    key = "glcm_rf"
    display_name = "Morphology + GLCM + Random Forest"
    feature_version = "glcm-morphology-v1"
    image_size = (128, 128)
    member_name = "Lee Siu Yee"
    task_description = "GLCM Texture Feature Extraction + Morphological Processing + Random Forest"
    report_section = "Section 2.3"

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = banana_mask(image, self.image_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)

        masked_gray = np.where(refined > 0, gray, 0).astype(np.uint8)
        quantised = (masked_gray // 16).astype(np.uint8)
        matrix = graycomatrix(
            quantised,
            distances=[1, 2, 4],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True,
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
