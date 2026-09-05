from __future__ import annotations

import numpy as np
import pytest

from src.live_assessment import SampledFrameAssessor, assess_image
from src.types import PredictionRecord


class StubPipeline:
    def __init__(self, label: str = "ripe", error: Exception | None = None) -> None:
        self.label = label
        self.error = error
        self.calls: list[tuple[np.ndarray, str]] = []

    def predict(self, image: np.ndarray, filename: str) -> PredictionRecord:
        self.calls.append((image, filename))
        if self.error:
            raise self.error
        return PredictionRecord(
            filename=filename,
            predicted_class=self.label,
            confidence=0.9,
            probabilities={"ripe": 0.9},
            algorithm="Stub",
            processing_time_ms=1.0,
        )


def models_for(*pipelines: StubPipeline):
    return {f"model_{index}": (pipeline, {}) for index, pipeline in enumerate(pipelines)}


def test_assess_image_reuses_pipeline_predict_contract():
    first, second = StubPipeline(), StubPipeline("unripe")
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    records = assess_image(models_for(first, second), image, "captured_banana.jpg")

    assert [record.predicted_class for record in records] == ["ripe", "unripe"]
    assert len(first.calls) == len(second.calls) == 1
    assert first.calls[0][0] is image and first.calls[0][1] == "captured_banana.jpg"
    assert second.calls[0][0] is image and second.calls[0][1] == "captured_banana.jpg"


def test_sampled_frame_assessor_skips_frames_until_interval_has_elapsed():
    pipeline = StubPipeline()
    assessor = SampledFrameAssessor(models_for(pipeline), interval_seconds=2.0)
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert assessor.submit(image, now=10.0)
    assert not assessor.submit(image, now=11.9)
    assert assessor.submit(image, now=12.0)
    assert len(pipeline.calls) == 2
    assert assessor.snapshot().records[0].filename == "live_camera_frame"
    assert np.array_equal(assessor.snapshot().image, image)


def test_sampled_frame_assessor_records_clean_error_and_continues():
    pipeline = StubPipeline(error=RuntimeError("model unavailable"))
    assessor = SampledFrameAssessor(models_for(pipeline), interval_seconds=1.0)

    assert assessor.submit(np.zeros((10, 10, 3), dtype=np.uint8), now=1.0)
    snapshot = assessor.snapshot()

    assert snapshot.records == ()
    assert snapshot.error == "Live assessment failed: model unavailable"
    assert snapshot.captured_at == 1.0


def test_sampled_frame_assessor_rejects_invalid_interval():
    with pytest.raises(ValueError, match="greater than zero"):
        SampledFrameAssessor({}, interval_seconds=0)


def test_sampled_frame_assessor_exposes_camera_errors_to_the_ui():
    assessor = SampledFrameAssessor({}, interval_seconds=1.0)

    assessor.record_error("Camera frame could not be read.", now=3.0)

    assert assessor.snapshot().error == "Camera frame could not be read."
    assert assessor.snapshot().captured_at == 3.0