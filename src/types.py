from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionRecord:
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    algorithm: str
    processing_time_ms: float
    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_explanation: bool = False) -> dict[str, Any]:
        result = asdict(self)
        if not include_explanation:
            result.pop("explanation", None)
            for key in (
                "blemish_ratio",
                "fruit_area_ratio",
                "fruit_area_px",
                "contour_area_px",
                "perimeter_px",
                "major_dimension_px",
                "minor_dimension_px",
            ):
                if key in self.explanation:
                    result[key] = self.explanation[key]
        for label, probability in self.probabilities.items():
            result[f"probability_{label}"] = probability
        result.pop("probabilities", None)
        return result

