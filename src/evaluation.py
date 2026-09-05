from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

from .constants import CLASS_NAMES


def _bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric,
    *,
    samples: int = 1_000,
    random_state: int = 42,
) -> list[float]:
    """Return a deterministic percentile 95% confidence interval."""
    if len(y_true) < 2 or samples <= 0:
        value = float(metric(y_true, y_pred))
        return [value, value]
    generator = np.random.default_rng(random_state)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = generator.integers(0, len(y_true), len(y_true))
        values[index] = metric(y_true[selected], y_pred[selected])
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    predicted_indices = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predicted_indices]
    true_indices = np.asarray([CLASS_NAMES.index(str(label)) for label in y_true])
    correct = predicted_indices == true_indices
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            error += float(np.mean(selected)) * abs(float(np.mean(correct[selected])) - float(np.mean(confidence[selected])))
    return error


def evaluate_predictions(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    bootstrap_samples: int = 1_000,
) -> dict[str, Any]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true_array, y_pred_array, labels=list(CLASS_NAMES))
    accuracy_metric = lambda truth, predicted: accuracy_score(truth, predicted)
    macro_f1_metric = lambda truth, predicted: f1_score(
        truth, predicted, labels=list(CLASS_NAMES), average="macro", zero_division=0
    )
    result = {
        "accuracy": float(accuracy_metric(y_true_array, y_pred_array)),
        "macro_f1": float(macro_f1_metric(y_true_array, y_pred_array)),
        "weighted_f1": float(f1_score(y_true_array, y_pred_array, labels=list(CLASS_NAMES), average="weighted", zero_division=0)),
        "accuracy_ci95": _bootstrap_interval(y_true_array, y_pred_array, accuracy_metric, samples=bootstrap_samples),
        "macro_f1_ci95": _bootstrap_interval(y_true_array, y_pred_array, macro_f1_metric, samples=bootstrap_samples),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.shape != (len(y_true_array), len(CLASS_NAMES)):
            raise ValueError("Probabilities must have one row per item and one column per configured class.")
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        one_hot = np.eye(len(CLASS_NAMES))[[CLASS_NAMES.index(str(label)) for label in y_true_array]]
        result.update(
            {
                "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
                "log_loss": float(log_loss(y_true_array, probabilities, labels=list(CLASS_NAMES))),
                "expected_calibration_error": expected_calibration_error(y_true_array, probabilities),
            }
        )
    return result


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidate models only on validation data to protect the test set."""
    return sorted(
        results,
        key=lambda item: (item["validation_metrics"]["macro_f1"], item["validation_metrics"]["accuracy"]),
        reverse=True,
    )

