"""Member B Task: K-Means Clustering in L*a*b* Colour Space + K-Nearest Neighbours.

Student: Lim Tze Chuen
Report Section: 2.2 K-Means Clustering with L*a*b* Colour Space and KNN Classifier
Algorithm Overview:
    - Preprocessing: Perceptual L*a*b* transformation and lightness separation.
    - Color Space: CIE L*a*b* colour model for illumination-invariant chromaticity clustering.
    - Clustering: K-Means (k=4) with K-Means++ initialisation to partition peel colour modes.
    - Feature Extraction: Sorted cluster centres, cluster pixel weights, global L*a*b* means and standard deviations.
    - Classification: Distance-weighted K-Nearest Neighbours (KNN, k=7) with StandardScaler.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import RANDOM_STATE
from src.pipelines.base import BaseBananaPipeline


class LabKMeansKNNPipeline(BaseBananaPipeline):
    key = "lab_knn"
    display_name = "L*a*b* K-Means + KNN"
    feature_version = "lab-kmeans4-shared-preprocess-v3"
    cluster_count = 4
    member_name = "Lim Tze Chuen"
    task_description = "K-Means Clustering with L*a*b* Colour Space + K-Nearest Neighbours"
    report_section = "Section 2.2"

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb, mask = self.preprocess(image)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = lab[mask > 0]
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
