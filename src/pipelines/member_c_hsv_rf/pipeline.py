"""Member C Task: HSV Colour Histogram + Random Forest Classifier.

Student: Chen Kai Jian
Report Section: 2.1 HSV Colour Histogram + Random Forest Classifier
Algorithm Overview:
    - Preprocessing: Standardised 224x224 RGB image normalization and foreground extraction.
    - Color Space: RGB to Hue-Saturation-Value (HSV) mapping.
    - Feature Extraction: 16-bin normalized marginal histograms per channel (H, S, V) concatenated into a 48-dimensional feature vector.
    - Classification: 200-tree Random Forest Ensemble with Gini Impurity criterion and balanced class weighting.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_STATE
from src.pipelines.base import BaseBananaPipeline


class HSVHistogramRandomForestPipeline(BaseBananaPipeline):
    key = "hsv_rf"
    display_name = "HSV Histogram + Random Forest"
    feature_version = "hsv-hist-16x3-shared-preprocess-v3"
    image_size = (224, 224)
    member_name = "Chen Kai Jian"
    task_description = "HSV Colour Histogram + Random Forest Classifier"
    report_section = "Section 2.1"

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb, mask = self.preprocess(image)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        pixels = hsv[mask > 0] if np.any(mask) else hsv.reshape(-1, 3)
        ranges = ((0, 180), (0, 256), (0, 256))
        histograms = [np.histogram(pixels[:, channel], bins=16, range=value_range)[0] for channel, value_range in enumerate(ranges)]
        features = np.concatenate(histograms).astype(np.float32)
        total = float(features.sum())
        return features / total if total > 0 else features

    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )
