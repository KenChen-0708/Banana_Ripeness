from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.video import VideoValidationError, annotate_frame, sample_video


class FakeCapture:
    def __init__(self, _path: str) -> None:
        self.frames = [np.full((48, 64, 3), index, dtype=np.uint8) for index in range(20)]
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_WIDTH: 64.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
            cv2.CAP_PROP_FRAME_COUNT: 20.0,
        }.get(prop, 0.0)

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        return None


def test_video_sampling_uses_requested_timestamps(monkeypatch):
    monkeypatch.setattr("src.video.cv2.VideoCapture", FakeCapture)
    sampled = sample_video(b"synthetic video", ".avi", sample_fps=2.0, max_frames=10)
    assert sampled.frame_indices == [0, 5, 10, 15]
    assert sampled.timestamps_seconds == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert sampled.duration_seconds == pytest.approx(2.0)
    assert all(frame.shape == (48, 64, 3) for frame in sampled.frames)


def test_video_validation_and_annotation():
    with pytest.raises(VideoValidationError, match="empty"):
        sample_video(b"", ".mp4")
    frame = np.full((120, 180, 3), 220, dtype=np.uint8)
    annotated = annotate_frame(frame, "ripe", 0.92, 0.08)
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, frame)
