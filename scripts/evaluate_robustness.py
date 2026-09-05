from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import DEFAULT_ARTIFACT_DIR
from src.dataset import load_split
from src.evaluation import evaluate_predictions
from src.model_store import discover_bundles
from src.preprocessing import decode_image
from src.robustness import PERTURBATIONS, apply_perturbation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model sensitivity to common imaging perturbations.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "fruit_dataset")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--pipeline", default=None, help="Pipeline key; defaults to the validation-selected model.")
    parser.add_argument("--limit-per-class", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads((args.artifacts / "evaluation_summary.json").read_text(encoding="utf-8"))
    pipeline_key = args.pipeline or summary["recommended_pipeline"]
    pipeline = discover_bundles(args.artifacts / "models")[pipeline_key][0]
    items = load_split(args.dataset, "test", args.limit_per_class)
    truth = [item.label for item in items]
    rows = []
    clean_accuracy = None
    for perturbation in PERTURBATIONS:
        predictions = []
        for index, item in enumerate(items):
            image = apply_perturbation(decode_image(item.path), perturbation, random_state=42 + index)
            predictions.append(pipeline.predict(image).predicted_class)
        metrics = evaluate_predictions(truth, predictions, bootstrap_samples=500)
        if clean_accuracy is None:
            clean_accuracy = metrics["accuracy"]
        rows.append(
            {
                "condition": perturbation,
                "sample_count": len(items),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy_ci95_low": metrics["accuracy_ci95"][0],
                "accuracy_ci95_high": metrics["accuracy_ci95"][1],
                "accuracy_drop_from_clean": clean_accuracy - metrics["accuracy"],
            }
        )
        print(f"{perturbation}: accuracy={metrics['accuracy']:.3f}, macro-F1={metrics['macro_f1']:.3f}")
    output_dir = args.artifacts / "robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{pipeline_key}_robustness.csv", index=False)
    payload = {
        "pipeline_key": pipeline_key,
        "selection_basis": "validation macro-F1",
        "test_subset_per_class": args.limit_per_class,
        "conditions": rows,
    }
    (output_dir / f"{pipeline_key}_robustness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
