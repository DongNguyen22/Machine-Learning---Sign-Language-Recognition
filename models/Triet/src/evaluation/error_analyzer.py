"""Confusion matrix and learning-curve visualization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

ASL_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def plot_normalized_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    output_path: Path | str,
    labels: Iterable[str] = ASL_LABELS,
    title: str = "Normalized Confusion Matrix",
) -> Path:
    label_names = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_sums != 0,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(15, 13))
    sns.heatmap(
        normalized,
        annot=False,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={"label": "Normalized fraction"},
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def extract_top_confusions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    top_k: int = 10,
    labels: Iterable[str] = ASL_LABELS,
) -> list[dict[str, object]]:
    label_names = list(labels)
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    index = {label: position for position, label in enumerate(label_names)}
    pairs: list[dict[str, object]] = []

    for true_index, true_label in enumerate(label_names):
        support = int(cm[true_index].sum())
        for pred_index, pred_label in enumerate(label_names):
            if true_index == pred_index:
                continue
            count = int(cm[true_index, pred_index])
            if count:
                pairs.append({
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                    "percentage": 100.0 * count / support if support else 0.0,
                })

    pairs.sort(
        key=lambda item: (-int(item["count"]), -float(item["percentage"]),
                          str(item["true_label"]), str(item["predicted_label"])),
    )
    return pairs[:top_k]


def plot_learning_curves(
    history_path: Path | str,
    output_dir: Path | str,
    experiment_name: str,
) -> list[Path]:
    path = Path(history_path)
    if not path.is_file():
        return []

    history = json.loads(path.read_text(encoding="utf-8"))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("train_acc", [])
    val_acc = history.get("val_acc", [])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if train_loss or val_loss:
        epochs = range(1, max(len(train_loss), len(val_loss)) + 1)
        figure, axis = plt.subplots(figsize=(8, 5))
        if train_loss:
            axis.plot(range(1, len(train_loss) + 1), train_loss, label="Train")
        if val_loss:
            axis.plot(range(1, len(val_loss) + 1), val_loss, label="Validation")
        axis.set_title(f"{experiment_name} - Loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.grid(alpha=0.25)
        loss_path = output / f"{experiment_name}_learning_loss.png"
        figure.tight_layout()
        figure.savefig(loss_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(loss_path)

    if train_acc or val_acc:
        figure, axis = plt.subplots(figsize=(8, 5))
        if train_acc:
            axis.plot(range(1, len(train_acc) + 1), train_acc, label="Train")
        if val_acc:
            axis.plot(range(1, len(val_acc) + 1), val_acc, label="Validation")
        axis.set_title(f"{experiment_name} - Accuracy")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.legend()
        axis.grid(alpha=0.25)
        acc_path = output / f"{experiment_name}_learning_accuracy.png"
        figure.tight_layout()
        figure.savefig(acc_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        paths.append(acc_path)
    return paths
