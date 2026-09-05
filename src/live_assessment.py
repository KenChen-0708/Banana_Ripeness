from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any, Mapping

import numpy as np

from .types import PredictionRecord


@dataclass(frozen=True, slots=True)
class LiveAssessmentSnapshot:
    """The latest complete sampled assessment, safe to read from the UI thread."""

    records: tuple[PredictionRecord, ...] = ()
    image: np.ndarray | None = None
    captured_at: float | None = None
    error: str | None = None


def assess_image(models: Mapping[str, tuple[Any, dict[str, Any]]], image: np.ndarray, filename: str) -> list[PredictionRecord]:
    """Run the already-loaded pipelines through their existing prediction contract."""
    return [pipeline.predict(image, filename) for pipeline, _ in models.values()]


class SampledFrameAssessor:
    """Runs existing pipelines on sampled frames and shares the latest result safely.

    ``submit`` is designed for a WebRTC callback thread. It intentionally has no
    Streamlit dependency and never renders UI from that thread.
    """

    def __init__(self, models: Mapping[str, tuple[Any, dict[str, Any]]], interval_seconds: float = 2.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("The sampling interval must be greater than zero.")
        self._models = models
        self._interval_seconds = interval_seconds
        self._lock = Lock()
        self._last_started_at = float("-inf")
        self._snapshot = LiveAssessmentSnapshot()

    def submit(self, image: np.ndarray, filename: str = "live_camera_frame", now: float | None = None) -> bool:
        """Assess one frame when due; return whether an assessment was started."""
        started_at = monotonic() if now is None else now
        with self._lock:
            if started_at - self._last_started_at < self._interval_seconds:
                return False
            self._last_started_at = started_at

        try:
            records = tuple(assess_image(self._models, image, filename))
        except Exception as exc:  # Camera callbacks must not terminate the video stream.
            with self._lock:
                self._snapshot = LiveAssessmentSnapshot(captured_at=started_at, error=f"Live assessment failed: {exc}")
            return True

        with self._lock:
            self._snapshot = LiveAssessmentSnapshot(records=records, image=image.copy(), captured_at=started_at)
        return True

    def snapshot(self) -> LiveAssessmentSnapshot:
        with self._lock:
            return self._snapshot

    def record_error(self, message: str, now: float | None = None) -> None:
        """Expose camera/callback failures to the UI without interrupting the stream."""
        with self._lock:
            self._snapshot = LiveAssessmentSnapshot(captured_at=monotonic() if now is None else now, error=message)