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
from src.video import process_video


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
        assert "quality_grade" in res.timeline_df.columns
    finally:
        Path(video_path).unlink(missing_ok=True)


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
