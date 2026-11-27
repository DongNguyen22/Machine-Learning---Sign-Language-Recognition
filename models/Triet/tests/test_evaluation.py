"""Tests for metrics, confusion analysis, learning curves, and FPS benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.evaluation.benchmark_fps import benchmark_model, model_size
from src.evaluation.error_analyzer import (
    extract_top_confusions,
    plot_learning_curves,
    plot_normalized_confusion_matrix,
)
from src.evaluation.metrics_calculator import calculate_metrics


def test_metrics_perfect_and_per_class_shape() -> None:
    labels = tuple("ABC")
    y_true = ["A", "B", "C", "A", "B", "C"]
    y_pred = y_true.copy()

    metrics = calculate_metrics(y_true, y_pred, labels)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert set(metrics["per_class"]) == set(labels)


def test_top_confusions_and_percentage() -> None:
    y_true = ["A", "A", "A", "B", "B", "C"]
    y_pred = ["A", "B", "B", "B", "C", "C"]

    top = extract_top_confusions(y_true, y_pred, top_k=10, labels=("A", "B", "C"))

    assert top[0]["true_label"] == "A"
    assert top[0]["predicted_label"] == "B"
    assert top[0]["count"] == 2
    assert top[0]["percentage"] == 200.0 / 3.0


def test_confusion_matrix_plot_saved_at_300_dpi(tmp_path: Path) -> None:
    output = tmp_path / "confusion.png"
    result = plot_normalized_confusion_matrix(
        ["A", "A", "B"],
        ["A", "B", "B"],
        output,
        labels=("A", "B"),
    )

    assert result == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_learning_curve_files(tmp_path: Path) -> None:
    history = tmp_path / "exp_history.json"
    history.write_text(
        json.dumps({
            "train_loss": [0.8, 0.5],
            "val_loss": [0.9, 0.6],
            "train_acc": [0.5, 0.8],
            "val_acc": [0.4, 0.7],
        }),
        encoding="utf-8",
    )

    files = plot_learning_curves(history, tmp_path, "exp_test")

    assert len(files) == 2
    assert all(path.is_file() for path in files)


def test_model_size_and_cpu_benchmark(tmp_path: Path) -> None:
    model_path = tmp_path / "dummy.joblib"
    model_path.write_bytes(b"12345678")
    size = model_size(model_path)

    assert size["bytes"] == 8.0
    assert size["kb"] > 0.0
    assert size["mb"] > 0.0

    model = KNeighborsClassifier(n_neighbors=1)
    x = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int32)
    model.fit(x, y)

    benchmark = benchmark_model(model, x[0], warmup=2, iterations=5)
    assert benchmark["latency_ms"] >= 0.0
    assert benchmark["fps"] >= 0.0
