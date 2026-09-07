from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR
from src.dataset import DatasetItem, assert_no_source_leakage, class_counts, load_split
from src.evaluation import evaluate_predictions, rank_results
from src.model_store import build_manifest, save_bundle
from src.pipelines import PIPELINE_CLASSES, BaseBananaPipeline, create_pipeline
from src.preprocessing import decode_image


PROGRESS_PREFIX = "__TRAINING_PROGRESS__ "


def emit_progress(progress: float, message: str) -> None:
    """Emit a machine-readable update while keeping normal CLI output readable."""
    payload = {"progress": min(max(progress, 0.0), 1.0), "message": message}
    print(PROGRESS_PREFIX + json.dumps(payload), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare banana-ripeness pipelines.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "fruit_dataset")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--pipelines", nargs="+", choices=list(PIPELINE_CLASSES), default=list(PIPELINE_CLASSES))
    parser.add_argument("--limit-per-class", type=int, default=None, help="Balanced development subset per split/class.")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def extract_features(
    pipeline: BaseBananaPipeline,
    items: list[DatasetItem],
    cache_path: Path,
    use_cache: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if (
            cached["feature_version"].item() == pipeline.feature_version
            and list(cached["paths"]) == [str(item.path) for item in items]
        ):
            if progress_callback:
                progress_callback(len(items), len(items))
            return cached["features"], cached["labels"]

    features: list[np.ndarray] = []
    labels: list[str] = []
    total = len(items)
    update_interval = max(1, total // 100)
    for index, item in enumerate(items, start=1):
        features.append(pipeline.extract_features(decode_image(item.path)))
        labels.append(item.label)
        if progress_callback and (index % update_interval == 0 or index == total):
            progress_callback(index, total)
        if index % 500 == 0 or index == total:
            print(f"  {item.split}: extracted {index}/{total}", flush=True)
    matrix = np.vstack(features)
    label_array = np.asarray(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=matrix,
        labels=label_array,
        paths=np.asarray([str(item.path) for item in items]),
        feature_version=np.asarray(pipeline.feature_version),
    )
    return matrix, label_array


def predict_features(pipeline: BaseBananaPipeline, features: np.ndarray) -> np.ndarray:
    return np.asarray(pipeline._require_model().predict(features))


def save_confusion_csv(metrics: dict, path: Path) -> None:
    frame = pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    frame.index.name = "actual"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def train_pipeline(
    key: str,
    splits: dict[str, list[DatasetItem]],
    artifact_dir: Path,
    use_cache: bool,
    pipeline_index: int,
    pipeline_count: int,
) -> dict:
    pipeline = create_pipeline(key)
    print(f"\nTraining {pipeline.display_name}")
    started = perf_counter()

    def update(local_progress: float, message: str) -> None:
        overall_progress = (pipeline_index + local_progress) / pipeline_count
        emit_progress(overall_progress, f"{pipeline.display_name}: {message}")

    update(0.0, "preparing features")
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split_index, (split, items) in enumerate(splits.items()):
        cache = artifact_dir / "cache" / f"{key}_{pipeline.feature_version}_{split}.npz"
        matrices[split], labels[split] = extract_features(
            pipeline,
            items,
            cache,
            use_cache,
            progress_callback=lambda done, total, split=split, split_index=split_index: update(
                (split_index + (done / max(total, 1))) * 0.2,
                f"extracting {split} features ({done}/{total})",
            ),
        )

    update(0.6, "fitting classifier")
    pipeline.fit_features(matrices["train"], labels["train"])
    update(0.8, "evaluating validation split")
    validation_metrics = evaluate_predictions(labels["valid"], predict_features(pipeline, matrices["valid"]))
    update(0.87, "evaluating held-out test split")
    test_metrics = evaluate_predictions(labels["test"], predict_features(pipeline, matrices["test"]))
    update(0.94, "saving model and metrics")
    manifest = build_manifest(pipeline, validation_metrics, test_metrics)
    save_bundle(pipeline, manifest, artifact_dir / "models" / f"{key}.joblib")
    save_confusion_csv(test_metrics, artifact_dir / "metrics" / f"{key}_confusion.csv")
    update(1.0, "complete")

    elapsed = perf_counter() - started
    print(
        f"  validation accuracy={validation_metrics['accuracy']:.4f}; "
        f"test accuracy={test_metrics['accuracy']:.4f}; macro-F1={test_metrics['macro_f1']:.4f}; "
        f"elapsed={elapsed:.1f}s"
    )
    return {
        "pipeline_key": key,
        "algorithm": pipeline.display_name,
        "feature_version": pipeline.feature_version,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    splits = {split: load_split(args.dataset, split, args.limit_per_class) for split in ("train", "valid", "test")}
    assert_no_source_leakage(splits)
    print("Dataset counts:")
    for split, items in splits.items():
        print(f"  {split}: {len(items)} {class_counts(items)}")

    results = [
        train_pipeline(key, splits, args.artifacts, not args.no_cache, index, len(args.pipelines))
        for index, key in enumerate(args.pipelines)
    ]
    ranked = rank_results(results)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "development_subset_per_class": args.limit_per_class,
        "split_counts": {split: class_counts(items) for split, items in splits.items()},
        "ranking_rule": "Descending test macro-F1, then descending test accuracy",
        "recommended_pipeline": ranked[0]["pipeline_key"],
        "results": ranked,
    }
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranking_frame = pd.DataFrame(
        [
            {
                "rank": index,
                "pipeline_key": item["pipeline_key"],
                "algorithm": item["algorithm"],
                "validation_accuracy": item["validation_metrics"]["accuracy"],
                "test_accuracy": item["test_metrics"]["accuracy"],
                "test_macro_f1": item["test_metrics"]["macro_f1"],
                "test_weighted_f1": item["test_metrics"]["weighted_f1"],
            }
            for index, item in enumerate(ranked, start=1)
        ]
    )
    ranking_frame.to_csv(args.artifacts / "model_ranking.csv", index=False)
    print(f"\nRecommended pipeline: {summary['recommended_pipeline']}")
    print(f"Artifacts written to: {args.artifacts.resolve()}")


if __name__ == "__main__":
    main()

