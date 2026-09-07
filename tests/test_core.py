from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.constants import CLASS_NAMES, PIPELINE_IMAGE_SIZE
from src.dataset import DatasetItem, assert_no_source_leakage
from src.evaluation import evaluate_predictions, rank_results
from src.exports import records_csv, records_pdf
from src.model_store import IncompatibleModelError, build_manifest, load_bundle, save_bundle, validate_manifest
from src.pipelines import PIPELINE_CLASSES, create_pipeline
from src.preprocessing import ImageValidationError, banana_mask, brown_spot_ratio, decode_image, prepare_pipeline_image
from src.types import PredictionRecord


@pytest.fixture
def synthetic_banana() -> np.ndarray:
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    yy, xx = np.ogrid[:240, :320]
    fruit = ((xx - 160) / 105) ** 2 + ((yy - 120) / 42) ** 2 <= 1
    image[fruit] = (235, 205, 35)
    spot = (xx - 195) ** 2 + (yy - 120) ** 2 <= 14**2
    image[spot] = (100, 58, 24)
    return image


def test_all_feature_extractors_are_deterministic(synthetic_banana):
    expected_sizes = {"hsv_svm": 14, "lab_knn": 22, "hsv_rf": 48, "glcm_rf": 14}
    for key, expected_size in expected_sizes.items():
        pipeline = create_pipeline(key)
        first = pipeline.extract_features(synthetic_banana)
        second = pipeline.extract_features(synthetic_banana)
        assert first.shape == (expected_size,)
        assert np.all(np.isfinite(first))
        assert np.allclose(first, second)


def test_all_pipelines_use_the_identical_shared_preprocessing_contract(synthetic_banana):
    expected_rgb, expected_mask = prepare_pipeline_image(synthetic_banana, PIPELINE_IMAGE_SIZE)

    assert expected_rgb.shape == (224, 224, 3)
    assert expected_rgb.dtype == np.uint8
    assert expected_mask.shape == (224, 224)
    assert expected_mask.dtype == np.uint8
    assert set(np.unique(expected_mask)).issubset({0, 255})

    for pipeline_class in PIPELINE_CLASSES.values():
        pipeline = pipeline_class()
        rgb, mask = pipeline.preprocess(synthetic_banana)
        assert pipeline.image_size == PIPELINE_IMAGE_SIZE
        assert np.array_equal(rgb, expected_rgb)
        assert np.array_equal(mask, expected_mask)


def test_hsv_histogram_is_normalised(synthetic_banana):
    features = create_pipeline("hsv_rf").extract_features(synthetic_banana)
    assert features.sum() == pytest.approx(1.0, abs=1e-6)


def test_mask_and_blemish_ratio_bounds(synthetic_banana):
    mask = banana_mask(synthetic_banana)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    ratio = brown_spot_ratio(synthetic_banana, mask)
    assert 0.0 <= ratio <= 1.0


def test_mask_follows_banana_on_dark_background():
    image = np.full((224, 224, 3), (65, 72, 80), dtype=np.uint8)
    yy, xx = np.ogrid[:224, :224]
    fruit = ((xx - 112) / 82) ** 2 + ((yy - 112) / 36) ** 2 <= 1
    image[fruit] = (245, 190, 45)
    mask = banana_mask(image)
    expected = fruit.astype(bool)
    predicted = cv_mask = mask > 0
    intersection = np.count_nonzero(predicted & expected)
    union = np.count_nonzero(predicted | expected)
    assert intersection / union > 0.85
    assert 0.10 < np.count_nonzero(cv_mask) / cv_mask.size < 0.50


def test_mask_does_not_treat_blank_frame_as_fruit():
    blank = np.full((224, 224, 3), 128, dtype=np.uint8)
    assert np.count_nonzero(banana_mask(blank)) == 0


def test_decode_rejects_non_image():
    with pytest.raises(ImageValidationError):
        decode_image(b"not an image")


def test_source_leakage_detection():
    splits = {
        "train": [DatasetItem(Path("train/ripe/sample_jpg.rf.abc.jpg"), "ripe", "train")],
        "test": [DatasetItem(Path("test/ripe/sample_jpg.rf.xyz.jpg"), "ripe", "test")],
    }
    with pytest.raises(ValueError, match="Source leakage"):
        assert_no_source_leakage(splits)


def test_metrics_and_ranking():
    truth = list(CLASS_NAMES)
    perfect = evaluate_predictions(truth, truth)
    imperfect = evaluate_predictions(truth, ["ripe"] * 4)
    ranked = rank_results(
        [
            {"pipeline_key": "weak", "test_metrics": imperfect},
            {"pipeline_key": "strong", "test_metrics": perfect},
        ]
    )
    assert perfect["accuracy"] == 1.0
    assert ranked[0]["pipeline_key"] == "strong"


def test_manifest_validation():
    pipeline = create_pipeline("hsv_rf")
    manifest = {
        "schema_version": 1,
        "pipeline_key": "hsv_rf",
        "feature_version": pipeline.feature_version,
        "image_size": list(pipeline.image_size),
        "class_order": list(CLASS_NAMES),
    }
    validate_manifest(manifest, pipeline)
    manifest["feature_version"] = "wrong"
    with pytest.raises(IncompatibleModelError):
        validate_manifest(manifest, pipeline)


def test_loading_bundle_rejects_a_preprocessing_version_mismatch(tmp_path, synthetic_banana):
    pipeline = create_pipeline("hsv_rf")
    pipeline.fit([synthetic_banana] * 4, CLASS_NAMES)
    manifest = build_manifest(pipeline, {}, {})
    model_path = tmp_path / "hsv_rf.joblib"
    save_bundle(pipeline, manifest, model_path)

    manifest["feature_version"] = "hsv-hist-16x3-grabcut-v2"
    model_path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    joblib.dump({"pipeline": pipeline, "manifest": manifest}, model_path)
    with pytest.raises(IncompatibleModelError, match="feature version"):
        load_bundle(model_path)


def test_exports_are_non_empty():
    record = PredictionRecord(
        filename="banana.jpg",
        predicted_class="ripe",
        confidence=0.91,
        probabilities={label: (0.91 if label == "ripe" else 0.03) for label in CLASS_NAMES},
        algorithm="Test model",
        processing_time_ms=12.5,
    )
    assert b"banana.jpg" in records_csv([record])
    assert records_pdf([record]).startswith(b"%PDF")

