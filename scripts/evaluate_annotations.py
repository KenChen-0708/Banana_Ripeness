from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import analyse_object, decode_image
from src.validation import bounding_box_iou, measurement_summary, segmentation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation, detection and blemish measurements against annotations.")
    parser.add_argument("annotations", type=Path, help="CSV following annotations/template.csv.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "annotation_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = pd.read_csv(args.annotations)
    required = {"image_path", "mask_path", "blemish_ratio", "bbox_x", "bbox_y", "bbox_width", "bbox_height"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {', '.join(sorted(missing))}")
    rows: list[dict[str, float | str]] = []
    for row in annotations.to_dict("records"):
        image_path = (args.annotations.parent / str(row["image_path"])).resolve()
        mask_path = (args.annotations.parent / str(row["mask_path"])).resolve()
        analysis = analyse_object(decode_image(image_path))
        reference_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if reference_mask is None:
            raise ValueError(f"Cannot read reference mask: {mask_path}")
        reference_mask = cv2.resize(reference_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_scores = segmentation_metrics(np.asarray(analysis["fruit_mask"]), reference_mask)
        bbox_score = bounding_box_iou(
            analysis["bounding_box"],
            (row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"]),
        )
        rows.append(
            {
                "image_path": str(image_path),
                **mask_scores,
                "bbox_iou": bbox_score,
                "predicted_blemish_ratio": float(analysis["blemish_ratio"]),
                "reference_blemish_ratio": float(row["blemish_ratio"]),
            }
        )
    result = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "per_image_metrics.csv", index=False)
    summary = {
        "sample_count": len(result),
        "mean_iou": float(result["iou"].mean()),
        "mean_dice": float(result["dice"].mean()),
        "mean_bbox_iou": float(result["bbox_iou"].mean()),
        "blemish_measurement": measurement_summary(result["predicted_blemish_ratio"], result["reference_blemish_ratio"]),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
