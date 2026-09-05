from __future__ import annotations

import numpy as np
import pytest

from src.pipelines import create_pipeline
from src.preprocessing import banana_mask, colour_masks


@pytest.fixture
def synthetic_banana() -> np.ndarray:
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    yy, xx = np.ogrid[:240, :320]
    fruit = ((xx - 160) / 105) ** 2 + ((yy - 120) / 42) ** 2 <= 1
    image[fruit] = (235, 205, 35)
    spot = (xx - 195) ** 2 + (yy - 120) ** 2 <= 14**2
    image[spot] = (100, 58, 24)
    return image


def test_hsv_svm_exposes_report_defined_brown_spot_ratio(synthetic_banana):
    pipeline = create_pipeline("hsv_svm")
    features = pipeline.extract_features(synthetic_banana)
    mask = banana_mask(synthetic_banana, pipeline.image_size)
    masks = colour_masks(synthetic_banana, mask=mask, size=pipeline.image_size)
    fruit_pixels = max(1, int(np.count_nonzero(mask)))
    expected_ratio = float(np.count_nonzero(masks["brown"])) / fruit_pixels

    assert features.shape == (14,)
    assert features[2] == pytest.approx(expected_ratio)

    explanation = pipeline.explain(synthetic_banana)
    assert explanation["brown_spot_ratio"] == pytest.approx(expected_ratio)
    assert 0.0 <= explanation["brown_spot_ratio"] <= 1.0
    assert 0.0 <= explanation["dark_ratio"] <= 1.0


def test_hsv_svm_brown_spot_ratio_increases_when_a_brown_spot_is_added(synthetic_banana):
    pipeline = create_pipeline("hsv_svm")
    without_spot = synthetic_banana.copy()
    yy, xx = np.ogrid[: without_spot.shape[0], : without_spot.shape[1]]
    spot = (xx - 195) ** 2 + (yy - 120) ** 2 <= 14**2
    without_spot[spot] = (235, 205, 35)

    assert pipeline.extract_features(synthetic_banana)[2] > pipeline.extract_features(without_spot)[2]