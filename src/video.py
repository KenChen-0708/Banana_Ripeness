from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.constants import CLASS_NAMES, DEFAULT_PIXELS_PER_CM
from src.pipelines.base import BaseBananaPipeline
from src.preprocessing import calibrate_fruit_geometry, determine_quality_grade, draw_annotated_geometry
from src.types import PredictionRecord


@dataclass
class VideoProcessingResult:
    total_frames_in_video: int
    frames_analyzed: int
    duration_sec: float
    fps: float
    timeline_df: pd.DataFrame
    sampled_frames: list[tuple[float, np.ndarray, PredictionRecord, dict[str, Any]]]
    class_counts: dict[str, int]
    grade_counts: dict[str, int]
    mean_confidence: float
    mean_blemish_ratio: float


def process_video(
    video_source: str | Path | bytes | bytearray,
    pipeline: BaseBananaPipeline,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    pixels_per_cm: float = DEFAULT_PIXELS_PER_CM,
) -> VideoProcessingResult:
    """Ingest a video stream or file, sampling frames to classify ripeness and measure quality."""
    temp_file_path = None
    if isinstance(video_source, (bytes, bytearray)):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_file.write(video_source)
        temp_file.flush()
        temp_file.close()
        video_path = temp_file.name
        temp_file_path = Path(video_path)
    else:
        video_path = str(video_source)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        raise ValueError("Could not open video file or video stream.")

    try:
        video_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_frames / video_fps if total_fps_valid(video_fps, total_frames) else 0.0

        step_interval = max(1, int(round(video_fps / max(0.1, sample_fps))))
        frame_idx = 0
        analyzed_count = 0

        timeline_records = []
        sampled_frames = []

        while capture.isOpened() and analyzed_count < max_frames:
            success, frame_bgr = capture.read()
            if not success:
                break

            if frame_idx % step_interval == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / video_fps

                filename = f"frame_{frame_idx:05d}_{timestamp:.2f}s"
                record = pipeline.predict(frame_rgb, filename=filename)
                geometry = calibrate_fruit_geometry(frame_rgb, pixels_per_cm=pixels_per_cm)
                grade = determine_quality_grade(record.predicted_class, geometry["blemish_ratio"])

                # Draw overlay for visual scrubber
                annotated = draw_annotated_geometry(frame_rgb, geometry)
                status_text = f"{record.predicted_class.upper()} ({record.confidence:.0%}) | {grade}"
                cv2.putText(
                    annotated,
                    status_text,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    status_text,
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (24, 74, 53),
                    1,
                    cv2.LINE_AA,
                )

                sampled_frames.append((timestamp, annotated, record, geometry))
                timeline_records.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp, 2),
                        "predicted_class": record.predicted_class,
                        "confidence": record.confidence,
                        "blemish_ratio": geometry["blemish_ratio"],
                        "blemish_percent": geometry["blemish_ratio"] * 100.0,
                        "quality_grade": grade,
                        "length_cm": geometry["physical_length_cm"],
                        "width_cm": geometry["physical_width_cm"],
                        "area_cm2": geometry["physical_area_cm2"],
                        "processing_time_ms": record.processing_time_ms,
                    }
                )
                analyzed_count += 1

            frame_idx += 1

    finally:
        capture.release()
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)

    timeline_df = pd.DataFrame(timeline_records)
    if timeline_df.empty:
        class_counts = {c: 0 for c in CLASS_NAMES}
        grade_counts = {}
        mean_conf = 0.0
        mean_blem = 0.0
    else:
        class_counts = timeline_df["predicted_class"].value_counts().to_dict()
        for c in CLASS_NAMES:
            class_counts.setdefault(c, 0)
        grade_counts = timeline_df["quality_grade"].value_counts().to_dict()
        mean_conf = float(timeline_df["confidence"].mean())
        mean_blem = float(timeline_df["blemish_ratio"].mean())

    return VideoProcessingResult(
        total_frames_in_video=total_frames,
        frames_analyzed=analyzed_count,
        duration_sec=duration_sec,
        fps=video_fps,
        timeline_df=timeline_df,
        sampled_frames=sampled_frames,
        class_counts=class_counts,
        grade_counts=grade_counts,
        mean_confidence=mean_conf,
        mean_blemish_ratio=mean_blem,
    )


def total_fps_valid(fps: float, total_frames: int) -> bool:
    return fps > 0 and total_frames > 0
