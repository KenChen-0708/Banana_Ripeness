from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from src.preprocessing import banana_mask, colour_masks
from src.types import PredictionRecord


class BaseBananaPipeline(ABC):
    key: str
    display_name: str
    feature_version = "1.0"
    image_size = (224, 224)

    def __init__(self) -> None:
        self.model: Any | None = None

    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def build_estimator(self) -> Any:
        raise NotImplementedError

    def extract_many(self, images: Iterable[np.ndarray]) -> np.ndarray:
        features = [self.extract_features(image) for image in images]
        if not features:
            raise ValueError("At least one image is required.")
        return np.vstack(features)

    def fit(self, images: Iterable[np.ndarray], labels: Iterable[str]) -> "BaseBananaPipeline":
        return self.fit_features(self.extract_many(images), labels)

    def fit_features(self, features: np.ndarray, labels: Iterable[str]) -> "BaseBananaPipeline":
        self.model = self.build_estimator()
        self.model.fit(features, np.asarray(list(labels)))
        return self

    def _require_model(self) -> Any:
        if self.model is None:
            raise RuntimeError(f"{self.display_name} has not been trained or loaded.")
        return self.model

    def predict_proba(self, image: np.ndarray) -> dict[str, float]:
        model = self._require_model()
        probabilities = model.predict_proba(self.extract_features(image).reshape(1, -1))[0]
        return {str(label): float(value) for label, value in zip(model.classes_, probabilities, strict=True)}

    def predict(self, image: np.ndarray, filename: str = "image") -> PredictionRecord:
        started = perf_counter()
        probabilities = self.predict_proba(image)
        predicted = max(probabilities, key=probabilities.get)
        elapsed_ms = (perf_counter() - started) * 1000
        return PredictionRecord(
            filename=filename,
            predicted_class=predicted,
            confidence=probabilities[predicted],
            probabilities=probabilities,
            algorithm=self.display_name,
            processing_time_ms=elapsed_ms,
            explanation=self.explain(image),
        )

    def explain(self, image: np.ndarray) -> dict[str, Any]:
        mask = banana_mask(image, self.image_size)
        masks = colour_masks(image, mask=mask, size=self.image_size)
        fruit_pixels = max(1, np.count_nonzero(mask))
        return {
            "fruit_mask": mask,
            "brown_mask": np.bitwise_or(masks["brown"], masks["dark"]),
            "green_ratio": float(np.count_nonzero(masks["green"])) / fruit_pixels,
            "yellow_ratio": float(np.count_nonzero(masks["yellow"])) / fruit_pixels,
        }

    def parameters(self) -> dict[str, Any]:
        estimator = self._require_model()
        return estimator.get_params(deep=False) if hasattr(estimator, "get_params") else {}

