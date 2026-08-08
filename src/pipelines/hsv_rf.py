from __future__ import annotations

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.constants import RANDOM_STATE
from src.preprocessing import resize_rgb

from .base import BaseBananaPipeline


class HSVHistogramRandomForestPipeline(BaseBananaPipeline):
    key = "hsv_rf"
    display_name = "HSV Histogram + Random Forest"
    feature_version = "hsv-hist-16x3-v1"
    image_size = (224, 224)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histograms = [
            cv2.calcHist([hsv], [0], None, [16], [0, 180]),
            cv2.calcHist([hsv], [1], None, [16], [0, 256]),
            cv2.calcHist([hsv], [2], None, [16], [0, 256]),
        ]
        features = np.concatenate(histograms).astype(np.float32).ravel()
        total = float(features.sum())
        return features / total if total > 0 else features

    def build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

