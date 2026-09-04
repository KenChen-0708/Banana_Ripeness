from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoValidationError(ValueError):
    """Raised when an uploaded video cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class SampledVideo:
    frames: list[np.ndarray]
    frame_indices: list[int]
    timestamps_seconds: list[float]
    source_fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float


def sample_video(
    payload: bytes,
    suffix: str,
    sample_fps: float = 2.0,
    max_frames: int = 300,
    max_duration_seconds: float = 300.0,
) -> SampledVideo:
    """Decode a bounded sequence of RGB frames at approximately ``sample_fps``.

    OpenCV requires a seekable video source, so the upload is held in a temporary
    file only for the duration of decoding and is removed in the ``finally`` block.
    """
    if not payload:
        raise VideoValidationError("The uploaded video is empty.")
    if len(payload) > 200 * 1024 * 1024:
        raise VideoValidationError("Video exceeds the 200 MB safety limit.")
    if not 0.1 <= sample_fps <= 30:
        raise VideoValidationError("Sampling rate must be between 0.1 and 30 frames per second.")
    if not 1 <= max_frames <= 2_000:
        raise VideoValidationError("Maximum analysed frames must be between 1 and 2,000.")

    temporary_path: Path | None = None
    capture: cv2.VideoCapture | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        capture = cv2.VideoCapture(str(temporary_path))
        if not capture.isOpened():
            raise VideoValidationError("The video could not be opened. Try MP4 (H.264/MPEG-4) or AVI.")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if width < 16 or height < 16:
            raise VideoValidationError("Video frames must be at least 16 x 16 pixels.")
        if width * height > 16_000_000:
            raise VideoValidationError("Video resolution exceeds the 16-megapixel safety limit.")

        reported_duration = total_frames / source_fps if total_frames else 0.0
        interval = 1.0 / sample_fps
        next_sample_time = 0.0
        frames: list[np.ndarray] = []
        indices: list[int] = []
        timestamps: list[float] = []
        frame_index = 0
        last_timestamp = 0.0
        while len(frames) < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            last_timestamp = timestamp
            if timestamp > max_duration_seconds:
                break
            if timestamp + (0.5 / source_fps) >= next_sample_time:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                indices.append(frame_index)
                timestamps.append(timestamp)
                next_sample_time += interval
            frame_index += 1
        if not frames:
            raise VideoValidationError("No decodable frames were found in the video.")
        duration = reported_duration if reported_duration > 0 else last_timestamp
        return SampledVideo(
            frames=frames,
            frame_indices=indices,
            timestamps_seconds=timestamps,
            source_fps=source_fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=duration,
        )
    finally:
        if capture is not None:
            capture.release()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def annotate_frame(
    frame: np.ndarray,
    predicted_class: str,
    confidence: float,
    blemish_ratio: float,
) -> np.ndarray:
    """Render a readable prediction overlay on an RGB video frame."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    scale = max(0.55, min(width, height) / 650.0)
    thickness = max(1, int(round(scale * 2)))
    panel_height = max(58, int(round(78 * scale)))
    cv2.rectangle(annotated, (0, 0), (width, panel_height), (7, 22, 20), -1)
    cv2.putText(
        annotated,
        f"{predicted_class.title()}  {confidence:.1%}",
        (14, max(25, int(31 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (125, 229, 142),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Blemished surface: {blemish_ratio:.1%}",
        (14, max(49, int(61 * scale))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale * 0.72,
        (240, 244, 242),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    return annotated
