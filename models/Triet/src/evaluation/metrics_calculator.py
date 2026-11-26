"""Classification metrics for ASL A-Z evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ASL_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def calculate_metrics(
    y_true: Sequence[str | int],
    y_pred: Sequence[str | int],
    labels: Iterable[str] = ASL_LABELS,
) -> dict[str, object]:
    label_names = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    per_class: dict[str, dict[str, float]] = {}

    for index, label in enumerate(label_names):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(label)] = {
            "accuracy": (tp + tn) / cm.sum() if cm.sum() else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[index, :].sum()),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0)
        ),
        "per_class": per_class,
    }


def per_class_dataframe(metrics: dict[str, object]) -> pd.DataFrame:
    per_class = metrics["per_class"]
    rows = [
        {"label": label, **values}
        for label, values in per_class.items()
    ]
    return pd.DataFrame(rows)
