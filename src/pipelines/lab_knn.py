from __future__ import annotations

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import RANDOM_STATE
from src.preprocessing import banana_mask, resize_rgb

from .base import BaseBananaPipeline


class LabKMeansKNNPipeline(BaseBananaPipeline):
    key = "lab_knn"
    display_name = "L*a*b* K-Means + KNN"
    feature_version = "lab-kmeans4-v1"
    image_size = (96, 96)
    cluster_count = 4

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mask = banana_mask(image, self.image_size) > 0
        pixels = lab[mask]
        if pixels.shape[0] < self.cluster_count:
            pixels = lab.reshape(-1, 3)
        if pixels.shape[0] > 2048:
            indices = np.linspace(0, pixels.shape[0] - 1, 2048, dtype=int)
            pixels = pixels[indices]

        cv2.setRNGSeed(RANDOM_STATE)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        _, labels, centres = cv2.kmeans(
            pixels,
            self.cluster_count,
            None,
            criteria,
            1,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.ravel(), minlength=self.cluster_count).astype(np.float32)
        weights = counts / max(1.0, counts.sum())
        order = np.argsort(centres[:, 0])
        centres = centres[order] / 255.0
        weights = weights[order]
        global_mean = pixels.mean(axis=0) / 255.0
        global_std = pixels.std(axis=0) / 255.0
        return np.asarray([*centres.ravel(), *weights, *global_mean, *global_std], dtype=np.float32)

    def build_estimator(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
            ]
        )

