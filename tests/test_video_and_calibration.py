from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.constants import CLASS_NAMES, DEFAULT_PIXELS_PER_CM
from src.exports import records_csv, records_dataframe, records_pdf
from src.pipelines import create_pipeline
from src.preprocessing import (
    apply_bilateral_filter,
    apply_clahe,
    apply_contrast_stretching,
    apply_gaussian_filter,
    apply_median_filter,
    apply_morphology,
    banana_mask,
    calibrate_fruit_geometry,
    determine_quality_grade,
    draw_annotated_geometry,
)
from src.types import PredictionRecord
from src.video import (
    VideoValidationError,
    analyse_frame,
    draw_live_overlay,
    live_preview_size,
    probe_video,
    process_video,
)


@pytest.fixture
def synthetic_fruit_image() -> np.ndarray:
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    yy, xx = np.ogrid[:240, :320]
    fruit = ((xx - 160) / 100) ** 2 + ((yy - 120) / 40) ** 2 <= 1
    image[fruit] = (235, 205, 35)
    spot = (xx - 190) ** 2 + (yy - 120) ** 2 <= 15**2
    image[spot] = (100, 58, 24)
    return image


def test_filters_output_shapes_and_types(synthetic_fruit_image):
    g = apply_gaussian_filter(synthetic_fruit_image, kernel_size=5, sigma=1.2)
    m = apply_median_filter(synthetic_fruit_image, kernel_size=5)
    b = apply_bilateral_filter(synthetic_fruit_image, d=7)
    cs = apply_contrast_stretching(synthetic_fruit_image)
    cl = apply_clahe(synthetic_fruit_image)

    for filtered in (g, m, b, cs, cl):
        assert filtered.shape == synthetic_fruit_image.shape
        assert filtered.dtype == np.uint8


def test_morphology_operations(synthetic_fruit_image):
    mask = banana_mask(synthetic_fruit_image)
    eroded = apply_morphology(mask, op="erode", kernel_size=5)
    dilated = apply_morphology(mask, op="dilate", kernel_size=5)
    closed = apply_morphology(mask, op="close", kernel_size=5)

    assert eroded.shape == mask.shape
    assert dilated.shape == mask.shape
    assert closed.shape == mask.shape
    assert np.count_nonzero(eroded) <= np.count_nonzero(mask)
    assert np.count_nonzero(dilated) >= np.count_nonzero(mask)


def test_spatial_calibration_and_annotation(synthetic_fruit_image):
    geom = calibrate_fruit_geometry(synthetic_fruit_image, pixels_per_cm=DEFAULT_PIXELS_PER_CM)

    assert geom["physical_length_cm"] > 0
    assert geom["physical_width_cm"] > 0
    assert geom["physical_area_cm2"] > 0
    assert 0.0 <= geom["blemish_ratio"] <= 1.0
    assert geom["quality_grade"] in {"Grade A (Premium)", "Grade B (Standard)", "Grade C (Processing)", "Reject / Rotten"}

    annotated = draw_annotated_geometry(synthetic_fruit_image, geom)
    assert annotated.shape == (224, 224, 3)
    assert annotated.dtype == np.uint8


def test_live_preview_preserves_camera_aspect_ratio(synthetic_fruit_image):
    assert live_preview_size(synthetic_fruit_image) == (640, 480)


def test_live_overlay_is_compact_and_keeps_frame_dimensions():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    lines = [
        "QUALITY: Grade A (Premium)",
        "Morphology + GLCM + Random Forest: RIPE 98%",
        "HSV Histogram + Random Forest: RIPE 97%",
        "HSV Segmentation + Brown Spot Rules: RIPE 96%",
        "L*a*b* K-Means + KNN: RIPE 95%",
    ]

    annotated = draw_live_overlay(image, lines)

    assert annotated.shape == image.shape
    assert np.any(annotated[:120] != image[:120])
    assert np.array_equal(annotated[140:], image[140:])


def test_live_analysis_can_return_a_widescreen_frame(synthetic_fruit_image):
    pipeline = create_pipeline("hsv_rf")
    pipeline.fit([synthetic_fruit_image] * 4, list(CLASS_NAMES))

    annotated, _, geometry, _ = analyse_frame(
        synthetic_fruit_image,
        pipeline,
        DEFAULT_PIXELS_PER_CM,
        output_size=(640, 480),
        show_status=False,
    )

    assert annotated.shape == (480, 640, 3)
    assert geometry["analysis_size"] == (640, 480)


def test_determine_quality_grade():
    assert determine_quality_grade("ripe", 0.02) == "Grade A (Premium)"
    assert determine_quality_grade("ripe", 0.08) == "Grade B (Standard)"
    assert determine_quality_grade("ripe", 0.22) == "Grade C (Processing)"
    assert determine_quality_grade("ripe", 0.35) == "Reject / Rotten"
    assert determine_quality_grade("rotten", 0.01) == "Reject / Rotten"


def test_video_processing_synthetic():
    pipeline = create_pipeline("hsv_rf")
    # Fit with synthetic data so predict works
    synthetic = np.full((224, 224, 3), 200, dtype=np.uint8)
    pipeline.fit([synthetic] * 4, list(CLASS_NAMES))

    # Create short 1-second video in temp file
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video.close()
    video_path = temp_video.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
    for _ in range(10):
        frame = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.circle(frame, (160, 120), 40, (0, 200, 200), -1)
        out.write(frame)
    out.release()

    try:
        res = process_video(video_path, pipeline, sample_fps=2.0, max_frames=5)
        assert res.frames_analyzed > 0
        assert not res.timeline_df.empty
        assert "blemish_percent" in res.timeline_df.columns
        assert "confidence_percent" in res.timeline_df.columns
        assert "source_frame_index" in res.timeline_df.columns
        assert "detection_valid" in res.timeline_df.columns
        assert "object_id" in res.timeline_df.columns
        assert "quality_grade" in res.timeline_df.columns
        assert res.effective_sample_fps > 0
        assert res.analysis_fps > 0
        assert res.mean_processing_time_ms >= 0
        assert res.metadata.width == 320
        assert res.metadata.height == 240
        assert res.annotated_video_bytes
    finally:
        Path(video_path).unlink(missing_ok=True)


def test_probe_video_rejects_invalid_bytes():
    with pytest.raises(VideoValidationError):
        probe_video(b"not a video", "broken.avi")


def test_video_frame_cap_is_reported():
    pipeline = create_pipeline("hsv_rf")
    synthetic = np.full((224, 224, 3), 200, dtype=np.uint8)
    pipeline.fit([synthetic] * 4, list(CLASS_NAMES))
    temp_video = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    temp_video.close()
    writer = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64))
    for _ in range(20):
        writer.write(np.full((64, 64, 3), 180, dtype=np.uint8))
    writer.release()
    try:
        result = process_video(temp_video.name, pipeline, sample_fps=10.0, max_frames=3)
        assert result.frames_analyzed == 3
        assert result.truncated
    finally:
        Path(temp_video.name).unlink(missing_ok=True)


def test_enhanced_pdf_and_dataframe_exports(synthetic_fruit_image):
    pipeline = create_pipeline("hsv_rf")
    pipeline.fit([synthetic_fruit_image] * 4, list(CLASS_NAMES))
    record = pipeline.predict(synthetic_fruit_image, "sample.jpg")

    df = records_dataframe([record])
    assert "quality_grade" in df.columns
    assert "blemish_percent" in df.columns
    assert "length_cm" in df.columns

    pdf_bytes = records_pdf([record])
    assert pdf_bytes.startswith(b"%PDF")
