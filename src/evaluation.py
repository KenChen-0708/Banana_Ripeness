from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .constants import CLASS_NAMES


def evaluate_predictions(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: (item["test_metrics"]["macro_f1"], item["test_metrics"]["accuracy"]),
        reverse=True,
    )

