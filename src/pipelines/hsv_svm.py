from __future__ import annotations

import cv2
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constants import RANDOM_STATE
from src.preprocessing import banana_mask, colour_masks, resize_rgb

from .base import BaseBananaPipeline


class HSVSegmentationSVMPipeline(BaseBananaPipeline):
    key = "hsv_svm"
    display_name = "HSV Segmentation + Brown Spot Ratio + SVM"
    feature_version = "hsv-ratios-statistics-v1"
    image_size = (224, 224)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        rgb = resize_rgb(image, self.image_size)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = banana_mask(image, self.image_size)
        masks = colour_masks(image, mask=mask, size=self.image_size)
        fruit = mask > 0
        count = max(1, int(np.count_nonzero(fruit)))
        ratios = [float(np.count_nonzero(masks[name])) / count for name in ("green", "yellow", "brown", "dark")]
        pixels = hsv[fruit]
        if not pixels.size:
            pixels = hsv.reshape(-1, 3)
        means = pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0])
        stds = pixels.std(axis=0) / np.array([180.0, 255.0, 255.0])
        percentiles = np.percentile(pixels[:, 2], [10, 50, 90]) / 255.0
        area_ratio = count / mask.size
        return np.asarray([*ratios, *means, *stds, *percentiles, area_ratio], dtype=np.float32)

    def build_estimator(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=10.0,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

