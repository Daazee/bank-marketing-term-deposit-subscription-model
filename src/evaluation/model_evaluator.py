"""Reusable classification metrics."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


class ModelEvaluator:
    """Calculate classification metrics from labels and optional probabilities."""

    @staticmethod
    def evaluate(y_true: Any, y_pred: Any, y_probability: Any | None = None) -> dict[str, Any]:
        """Return common metrics; probability metrics are omitted if unavailable."""
        scores: dict[str, Any] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, zero_division=0),
        }
        if y_probability is not None:
            probability = np.asarray(y_probability)
            if probability.ndim == 2:
                probability = probability[:, 1]
            scores["roc_auc"] = roc_auc_score(y_true, probability)
            scores["pr_auc"] = average_precision_score(y_true, probability)
        return scores

    @staticmethod
    def predict_at_threshold(probability: Any, threshold) -> np.ndarray:
        """Convert positive-class probabilities to labels at a chosen threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        values = np.asarray(probability)
        if values.ndim == 2:
            values = values[:, 1]
        return (values >= threshold).astype(int)
