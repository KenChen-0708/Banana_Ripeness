from __future__ import annotations

from typing import Iterable

import numpy as np


def segmentation_metrics(predicted_mask: np.ndarray, reference_mask: np.ndarray) -> dict[str, float]:
    """Compute binary-mask overlap metrics against a human-labelled reference."""
    predicted = np.asarray(predicted_mask) > 0
    reference = np.asarray(reference_mask) > 0
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference masks must have identical dimensions.")
    true_positive = int(np.count_nonzero(predicted & reference))
    false_positive = int(np.count_nonzero(predicted & ~reference))
    false_negative = int(np.count_nonzero(~predicted & reference))
    union = true_positive + false_positive + false_negative
    predicted_count = true_positive + false_positive
    reference_count = true_positive + false_negative
    return {
        "iou": true_positive / max(1, union),
        "dice": (2 * true_positive) / max(1, predicted_count + reference_count),
        "precision": true_positive / max(1, predicted_count),
        "recall": true_positive / max(1, reference_count),
    }


def bounding_box_iou(predicted: Iterable[float], reference: Iterable[float]) -> float:
    """Compute IoU for (x, y, width, height) bounding boxes."""
    px, py, pw, ph = (float(value) for value in predicted)
    rx, ry, rw, rh = (float(value) for value in reference)
    intersection_width = max(0.0, min(px + pw, rx + rw) - max(px, rx))
    intersection_height = max(0.0, min(py + ph, ry + rh) - max(py, ry))
    intersection = intersection_width * intersection_height
    union = max(0.0, pw * ph) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0 else 0.0


def measurement_summary(predicted: Iterable[float], reference: Iterable[float]) -> dict[str, float]:
    predicted_array = np.asarray(list(predicted), dtype=np.float64)
    reference_array = np.asarray(list(reference), dtype=np.float64)
    if predicted_array.shape != reference_array.shape or predicted_array.size == 0:
        raise ValueError("Predicted and reference measurement vectors must have the same non-zero length.")
    differences = predicted_array - reference_array
    correlation = float(np.corrcoef(predicted_array, reference_array)[0, 1]) if predicted_array.size > 1 else float("nan")
    return {
        "sample_count": int(predicted_array.size),
        "mae": float(np.mean(np.abs(differences))),
        "rmse": float(np.sqrt(np.mean(differences**2))),
        "bias": float(np.mean(differences)),
        "pearson_r": correlation,
    }
