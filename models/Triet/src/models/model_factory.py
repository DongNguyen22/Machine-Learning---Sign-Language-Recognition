"""Experiment model factory."""

from __future__ import annotations

from typing import Any

from src.models.mlp_classifier import build_baseline_mlp, build_optimized_mlp
from src.models.traditional_ml import create_knn, create_random_forest, create_svm


def create_model(
    experiment_name: str,
    input_dim: int,
    num_classes: int = 26,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0,
    x_train: object | None = None,
) -> Any:
    name = experiment_name.strip().lower()
    if name == "exp1_knn":
        return create_knn()
    if name == "exp1_mlp_base":
        return build_baseline_mlp(
            input_dim, num_classes, learning_rate, label_smoothing, x_train
        )
    if name == "exp2_mlp_geo":
        return build_optimized_mlp(
            input_dim, num_classes, learning_rate, label_smoothing, x_train
        )
    if name == "exp3_mlp_opt":
        return build_optimized_mlp(
            input_dim, num_classes, learning_rate, label_smoothing, x_train
        )
    if name == "exp3_svm_opt":
        return create_svm()
    if name == "exp1_random_forest":
        return create_random_forest()
    raise ValueError(f"Tên thí nghiệm không được hỗ trợ: {experiment_name}")
