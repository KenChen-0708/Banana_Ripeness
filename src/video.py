from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.constants import CLASS_NAMES, DEFAULT_PIXELS_PER_CM
from src.pipelines.base import BaseBananaPipeline
from src.preprocessing import calibrate_fruit_geometry, determine_quality_grade, draw_annotated_geometry
from src.types import PredictionRecord


class VideoValidationError(ValueError):
    """Raised when a video cannot be decoded or contains no usable frames."""


@dataclass(frozen=True)
class VideoMetadata:
    total_frames: int
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str


@dataclass
class VideoProcessingResult:
    total_frames_in_video: int
    frames_analyzed: int
    duration_sec: float
    fps: float
    effective_sample_fps: float
    timeline_df: pd.DataFrame
    sampled_frames: list[tuple[float, np.ndarray, PredictionRecord, dict[str, Any]]]
    class_counts: dict[str, int]
    grade_counts: dict[str, int]
    mean_confidence: float
    mean_blemish_ratio: float
    mean_processing_time_ms: float
    analysis_fps: float
    annotated_video_bytes: bytes | None
    annotated_video_mime: str | None
    truncated: bool
    metadata: VideoMetadata


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in {".mp4", ".avi", ".mov", ".mkv"} else ".mp4"


def _codec_name(value: float) -> str:
    integer = int(value)
    codec = "".join(chr((integer >> (8 * i)) & 0xFF) for i in range(4))
    return "".join(c for c in codec if c.isprintable()).strip() or "Unknown"


def _metadata(capture: cv2.VideoCapture) -> VideoMetadata:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    return VideoMetadata(
        total_frames=total,
        duration_sec=total / fps if total else 0.0,
        fps=fps,
        width=max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)),
        height=max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)),
        codec=_codec_name(capture.get(cv2.CAP_PROP_FOURCC)),
    )


def _materialise_video(
    video_source: str | Path | bytes | bytearray, source_filename: str | None
) -> tuple[Path, bool]:
    if not isinstance(video_source, (bytes, bytearray)):
        return Path(video_source), False
    with tempfile.NamedTemporaryFile(suffix=_safe_suffix(source_filename), delete=False) as stream:
        stream.write(video_source)
        return Path(stream.name), True


def probe_video(video_source: str | Path | bytes | bytearray, source_filename: str | None = None) -> VideoMetadata:
    """Read metadata and verify that the first frame is decodable."""
    path, temporary = _materialise_video(video_source, source_filename)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoValidationError("The video could not be opened. Check its codec or try MP4/H.264.")
        metadata = _metadata(capture)
        success, frame = capture.read()
        if not success or frame is None or frame.size == 0:
            raise VideoValidationError("The video contains no decodable frames.")
        return metadata
    finally:
        capture.release()
        if temporary:
            path.unlink(missing_ok=True)


def _detection_is_valid(geometry: dict[str, Any]) -> bool:
    width, height = geometry.get("analysis_size", (224, 224))
    area_ratio = float(geometry.get("pixel_area", 0.0)) / max(1.0, float(width * height))
    return geometry.get("contour") is not None and 0.015 <= area_ratio <= 0.92


def live_preview_size(frame_rgb: np.ndarray, target_width: int = 640) -> tuple[int, int]:
    """Return an aspect-ratio-preserving size suitable for a browser camera preview."""
    height, width = frame_rgb.shape[:2]
    if width <= 0 or height <= 0:
        return target_width, target_width
    return target_width, max(1, round(height * target_width / width))


def draw_live_overlay(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw compact, width-fitted live results without obscuring most of the frame."""
    if not lines:
        return image

    annotated = image.copy()
    height, width = annotated.shape[:2]
    margin = max(6, round(width * 0.0125))
    line_height = max(16, round(height * 0.045))
    panel_height = min(height, margin * 2 + line_height * len(lines))
    panel = annotated[:panel_height].copy()
    panel[:] = (12, 28, 23)
    annotated[:panel_height] = cv2.addWeighted(
        annotated[:panel_height], 0.25, panel, 0.75, 0
    )

    max_text_width = max(1, width - margin * 2)
    for index, text in enumerate(lines):
        font_scale = min(0.55, max(0.34, width / 1400.0))
        while font_scale > 0.28:
            (text_width, _), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            if text_width <= max_text_width:
                break
            font_scale -= 0.02
        baseline_y = margin + line_height * index + round(line_height * 0.72)
        cv2.putText(
            annotated,
            text,
            (margin, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 250, 247),
            1,
            cv2.LINE_AA,
        )
    return annotated


def analyse_frame(
    frame_rgb: np.ndarray,
    pipeline: BaseBananaPipeline,
    pixels_per_cm: float,
    confidence_threshold: float = 0.55,
    filename: str = "camera_frame",
    output_size: tuple[int, int] = (224, 224),
    show_status: bool = True,
) -> tuple[np.ndarray, PredictionRecord, dict[str, Any], str]:
    """Analyse and annotate one RGB frame for file or camera use."""
    record = pipeline.predict(frame_rgb, filename=filename)
    geometry = calibrate_fruit_geometry(
        frame_rgb, pixels_per_cm=pixels_per_cm, size=output_size
    )
    valid = _detection_is_valid(geometry)
    grade = determine_quality_grade(record.predicted_class, geometry["blemish_ratio"]) if valid else "No valid fruit detected"
    label = record.predicted_class if record.confidence >= confidence_threshold else "uncertain"
    geometry.update(detection_valid=valid, quality_grade=grade)
    annotated = (
        draw_annotated_geometry(frame_rgb, geometry, size=output_size)
        if valid
        else cv2.resize(frame_rgb, output_size, interpolation=cv2.INTER_AREA)
    )
    if show_status:
        status = f"{label.upper()} ({record.confidence:.0%}) | {grade}"
        annotated = draw_live_overlay(annotated, [status])
    return annotated, record, geometry, label


def process_video(
    video_source: str | Path | bytes | bytearray,
    pipeline: BaseBananaPipeline,
    sample_fps: float = 2.0,
    max_frames: int = 120,
    pixels_per_cm: float = DEFAULT_PIXELS_PER_CM,
    source_filename: str | None = None,
    confidence_threshold: float = 0.55,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoProcessingResult:
    """Analyse sampled frames from a video file and return auditable outputs."""
    video_path, temporary = _materialise_video(video_source, source_filename)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        if temporary:
            video_path.unlink(missing_ok=True)
        raise VideoValidationError("The video could not be opened. Check its codec or try MP4/H.264.")

    metadata = _metadata(capture)
    step_interval = max(1, int(round(metadata.fps / max(0.1, sample_fps))))
    effective_fps = metadata.fps / step_interval
    expected = min(max_frames, int(np.ceil(metadata.total_frames / step_interval))) if metadata.total_frames else max_frames
    output_path: Path | None = None
    writer: cv2.VideoWriter | None = None
    timeline: list[dict[str, Any]] = []
    samples: list[tuple[float, np.ndarray, PredictionRecord, dict[str, Any]]] = []
    tracks: dict[int, tuple[int, int]] = {}
    next_track_id = 1
    frame_idx = 0
    started = perf_counter()

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
            output_path = Path(output.name)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (224, 224))
        while capture.isOpened() and len(timeline) < max_frames:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_idx % step_interval:
                frame_idx += 1
                continue
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = timestamp_ms / 1000.0 if timestamp_ms > 0 else frame_idx / metadata.fps
            annotated, record, geometry, label = analyse_frame(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), pipeline, pixels_per_cm, confidence_threshold,
                filename=f"frame_{frame_idx:06d}_{timestamp:.2f}s",
            )
            object_id: int | None = None
            if geometry["detection_valid"]:
                centroid = geometry["centroid"]
                matches = [(track_id, float(np.linalg.norm(np.subtract(centroid, point)))) for track_id, point in tracks.items()]
                if matches and min(matches, key=lambda item: item[1])[1] <= 55.0:
                    object_id = min(matches, key=lambda item: item[1])[0]
                else:
                    object_id, next_track_id = next_track_id, next_track_id + 1
                tracks[object_id] = centroid
            geometry["object_id"] = object_id
            samples.append((timestamp, annotated, record, geometry))
            if writer.isOpened():
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            timeline.append({
                "source_frame_index": frame_idx, "timestamp_sec": round(timestamp, 3), "object_id": object_id,
                "detection_valid": geometry["detection_valid"], "predicted_class": record.predicted_class,
                "display_label": label, "confidence": record.confidence, "confidence_percent": record.confidence * 100.0,
                "blemish_ratio": geometry["blemish_ratio"], "blemish_percent": geometry["blemish_ratio"] * 100.0,
                "quality_grade": geometry["quality_grade"],
                "length_cm": geometry["physical_length_cm"] if geometry["detection_valid"] else np.nan,
                "width_cm": geometry["physical_width_cm"] if geometry["detection_valid"] else np.nan,
                "area_cm2": geometry["physical_area_cm2"] if geometry["detection_valid"] else np.nan,
                "processing_time_ms": record.processing_time_ms,
            })
            if progress_callback:
                progress_callback(len(timeline), max(1, expected))
            frame_idx += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temporary:
            video_path.unlink(missing_ok=True)

    if not timeline:
        if output_path:
            output_path.unlink(missing_ok=True)
        raise VideoValidationError("No frames could be analysed from this video.")
    elapsed = max(perf_counter() - started, 1e-9)
    data = pd.DataFrame(timeline)
    valid = data[data["detection_valid"]]
    class_counts = valid["display_label"].value_counts().to_dict()
    for name in (*CLASS_NAMES, "uncertain"):
        class_counts.setdefault(name, 0)
    grade_counts = valid["quality_grade"].value_counts().to_dict()
    annotated_bytes = output_path.read_bytes() if output_path and output_path.exists() and output_path.stat().st_size else None
    if output_path:
        output_path.unlink(missing_ok=True)
    return VideoProcessingResult(
        metadata.total_frames, len(data), metadata.duration_sec, metadata.fps, effective_fps, data, samples,
        class_counts, grade_counts, float(valid["confidence"].mean()) if not valid.empty else 0.0,
        float(valid["blemish_ratio"].mean()) if not valid.empty else 0.0,
        float(data["processing_time_ms"].mean()), len(data) / elapsed, annotated_bytes,
        "video/mp4" if annotated_bytes else None,
        metadata.total_frames > 0 and frame_idx < metadata.total_frames, metadata,
    )


def total_fps_valid(fps: float, total_frames: int) -> bool:
    return fps > 0 and total_frames > 0
