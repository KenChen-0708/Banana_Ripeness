from __future__ import annotations

import json
from pathlib import Path

from src.model_store import discover_bundles

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_full_benchmark_artifacts_and_reference_accuracy():
    summary = json.loads((PROJECT_ROOT / "artifacts" / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary["development_subset_per_class"] is None
    assert {split: sum(counts.values()) for split, counts in summary["split_counts"].items()} == {
        "train": 11_793,
        "valid": 1_123,
        "test": 562,
    }
    assert summary["recommended_pipeline"] == "hsv_rf"
    hsv_result = next(item for item in summary["results"] if item["pipeline_key"] == "hsv_rf")
    assert abs(hsv_result["test_metrics"]["accuracy"] - 0.9626334519572953) <= 0.005
    assert set(discover_bundles(PROJECT_ROOT / "artifacts" / "models")) == {
        "hsv_svm",
        "lab_knn",
        "hsv_rf",
        "glcm_rf",
    }

