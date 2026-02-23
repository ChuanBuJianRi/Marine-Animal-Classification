"""Accuracy, macro F1, and per-class metrics."""
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels))


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[np.ndarray] = None,
) -> dict[int, dict[str, float]]:
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        int(lab): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
        for i, lab in enumerate(labels)
    }


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None,
) -> dict:
    return classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
