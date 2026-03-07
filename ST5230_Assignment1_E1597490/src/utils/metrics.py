"""Evaluation metrics shared across experiments."""

import math
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_perplexity(avg_nll: float) -> float:
    """Convert average negative log-likelihood (nats) to perplexity.

    Args:
        avg_nll: Average negative log-likelihood per token (nats).

    Returns:
        Perplexity value.
    """
    return math.exp(avg_nll)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and macro F1 for a classification task.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.

    Returns:
        Dictionary with keys: accuracy, macro_f1, weighted_f1.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> np.ndarray:
    """Return confusion matrix aligned to the given label order.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        labels: Ordered list of class indices.

    Returns:
        Confusion matrix as a 2-D NumPy array.
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: list
) -> str:
    """Return a formatted sklearn classification report string.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        target_names: Human-readable class names.

    Returns:
        Classification report string.
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
