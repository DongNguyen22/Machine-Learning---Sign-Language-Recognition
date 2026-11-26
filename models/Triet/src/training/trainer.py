"""Training utilities for MLP and traditional ASL classifiers."""

from __future__ import annotations

import json
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.models.model_factory import create_model
from src.models.mlp_classifier import prepare_targets
from src.training.augmentation import augment_landmark_samples
from src.training.callbacks import create_training_callbacks


def load_feature_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Tập dữ liệu trống: {path}")
    if "label" not in frame.columns:
        raise ValueError(f"Thiếu cột label trong: {path}")
    feature_columns = sorted(
        (column for column in frame.columns if column.startswith("feature_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not feature_columns:
        raise ValueError(f"Thiếu cột feature trong: {path}")
    return frame


def _extract_xy(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = sorted(
        (column for column in frame.columns if column.startswith("feature_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = frame["label"].astype(str).to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Dữ liệu huấn luyện chứa NaN hoặc Inf.")
    return x, labels


def encode_labels(labels: np.ndarray, label_to_index: dict[str, int]) -> np.ndarray:
    try:
        return np.asarray(
            [label_to_index[str(label).upper()] for label in labels],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(f"Nhãn không nằm trong label map: {exc.args[0]}") from exc


def prepare_calibration_samples(
    x: np.ndarray,
    y: np.ndarray,
    minimum_per_class: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample only sparse present classes for calibrated 3-fold training."""
    unique, counts = np.unique(y, return_counts=True)
    extra_x: list[np.ndarray] = []
    extra_y: list[np.ndarray] = []
    for label, count in zip(unique, counts):
        count = int(count)
        if count >= minimum_per_class:
            continue
        class_rows = x[y == int(label)]
        needed = minimum_per_class - count
        extra_x.append(
            np.asarray(
                np.resize(class_rows, (needed, class_rows.shape[1])).copy(),
                dtype=np.float32,
            )
        )
        extra_y.append(
            np.full(needed, int(label), dtype=np.int32)
        )
    if not extra_x:
        return x.copy(), y.copy()
    return (
        np.concatenate((x, *extra_x), axis=0).astype(np.float32),
        np.concatenate((y, *extra_y), axis=0).astype(np.int32),
    )


def save_history(history: keras.callbacks.History, output_path: Path) -> Path:
    history_data = {
        "train_loss": [float(value) for value in history.history.get("loss", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "train_acc": [float(value) for value in history.history.get("accuracy", [])],
        "val_acc": [float(value) for value in history.history.get("val_accuracy", [])],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(history_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def export_onnx(keras_path: Path, onnx_path: Path) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = keras.models.load_model(keras_path)
    with tempfile.TemporaryDirectory(prefix="asl_saved_model_") as temp_dir:
        saved_model_dir = Path(temp_dir) / "saved_model"
        model.export(saved_model_dir)
        command = [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--saved-model",
            str(saved_model_dir),
            "--output",
            str(onnx_path),
            "--opset",
            "15",
        ]
        subprocess.run(command, check=True)
    return onnx_path


def dataset_file_stats(path: Path | str) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.is_file():
        return {
            "path": str(file_path),
            "samples": 0,
            "sha256": None,
            "created_at": None,
        }
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    frame = pd.read_parquet(file_path)
    return {
        "path": str(file_path),
        "samples": int(len(frame)),
        "sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def train_keras_model(
    experiment_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    label_to_index: dict[str, int],
    output_dir: Path,
    checkpoint_dir: Path,
    history_path: Path,
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    augment_train: bool = True,
    label_smoothing: float = 0.0,
    train_dataset_path: Path | str | None = None,
    val_dataset_path: Path | str | None = None,
) -> tuple[keras.Model, dict[str, object]]:
    train_x, train_labels = _extract_xy(train_frame)
    val_x, val_labels = _extract_xy(val_frame)
    train_y = encode_labels(train_labels, label_to_index)
    val_y = encode_labels(val_labels, label_to_index)

    if augment_train:
        train_x, train_y_aug = augment_landmark_samples(
            train_x,
            train_y,
            feature_mode="63d" if train_x.shape[1] == 63 else "91d",
            random_seed=seed,
        )
        train_y = train_y_aug

    model = create_model(
        experiment_name,
        input_dim=train_x.shape[1],
        num_classes=len(label_to_index),
        learning_rate=learning_rate,
        x_train=train_x,
        label_smoothing=label_smoothing,
    )
    if not isinstance(model, keras.Model):
        raise TypeError("Thí nghiệm này không phải mô hình Keras.")

    checkpoint_path = checkpoint_dir / f"{experiment_name}.keras"
    callbacks = create_training_callbacks(checkpoint_path)
    fit_train_y = prepare_targets(
        train_y,
        num_classes=len(label_to_index),
        label_smoothing=label_smoothing,
    )
    fit_val_y = prepare_targets(
        val_y,
        num_classes=len(label_to_index),
        label_smoothing=label_smoothing,
    )
    history = model.fit(
        train_x,
        fit_train_y,
        validation_data=(val_x, fit_val_y),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    save_history(history, history_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / f"{experiment_name}.keras"
    model.save(keras_path)
    onnx_path = output_dir / f"{experiment_name}.onnx"
    export_onnx(keras_path, onnx_path)
    val_losses = [float(value) for value in history.history.get("val_loss", [])]
    val_accuracies = [
        float(value) for value in history.history.get("val_accuracy", [])
    ]
    if val_losses:
        best_index = int(np.argmin(val_losses))
        best_val_loss = val_losses[best_index]
        best_val_accuracy = (
            val_accuracies[best_index]
            if best_index < len(val_accuracies)
            else float("nan")
        )
    else:
        best_val_loss = 0.0
        best_val_accuracy = 0.0

    train_stats = dataset_file_stats(
        train_dataset_path if train_dataset_path is not None else ""
    )
    val_stats = dataset_file_stats(
        val_dataset_path if val_dataset_path is not None else ""
    )
    metadata_path = output_dir / f"{experiment_name}_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "experiment_name": experiment_name,
            "model_type": "mlp",
            "model_path": str(keras_path),
            "onnx_path": str(onnx_path),
            "scaler_path": None,
            "input_dim": int(train_x.shape[1]),
            "num_classes": len(label_to_index),
            "label_smoothing": float(label_smoothing),
            "random_seed": int(seed),
            "dataset_stats": {
                "train": train_stats,
                "val": val_stats,
            },
            "best_val_loss": float(best_val_loss),
            "best_val_accuracy": float(best_val_accuracy),
            "label_to_index": label_to_index,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics: dict[str, object] = {
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
    }
    return model, metrics


def train_traditional_model(
    experiment_name: str,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    label_to_index: dict[str, int],
    output_dir: Path,
    seed: int = 42,
    train_dataset_path: Path | str | None = None,
    val_dataset_path: Path | str | None = None,
) -> tuple[Any, dict[str, object]]:
    train_x, train_labels = _extract_xy(train_frame)
    val_x, val_labels = _extract_xy(val_frame)
    train_y = encode_labels(train_labels, label_to_index)
    val_y = encode_labels(val_labels, label_to_index)
    model = create_model(experiment_name, input_dim=train_x.shape[1])
    if experiment_name == "exp3_svm_opt":
        train_x, train_y = prepare_calibration_samples(
            train_x,
            train_y,
            minimum_per_class=3,
        )
        model.named_steps["svm"].cv = 3
    model.fit(train_x, train_y)

    train_acc = float(model.score(train_x, train_y))
    val_acc = float(model.score(val_x, val_y))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{experiment_name}.joblib"
    joblib.dump(model, model_path)
    metadata_path = output_dir / f"{experiment_name}_metadata.json"
    metadata_path.write_text(
        json.dumps({
            "experiment_name": experiment_name,
            "model_type": "sklearn_pipeline",
            "model_path": str(model_path),
            "onnx_path": None,
            "scaler_path": str(model_path),
            "input_dim": int(train_x.shape[1]),
            "num_classes": len(label_to_index),
            "label_smoothing": 0.0,
            "random_seed": int(seed),
            "dataset_stats": {
                "train": dataset_file_stats(
                    train_dataset_path if train_dataset_path is not None else ""
                ),
                "val": dataset_file_stats(
                    val_dataset_path if val_dataset_path is not None else ""
                ),
            },
            "best_val_loss": None,
            "best_val_accuracy": float(val_acc),
            "label_to_index": label_to_index,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "model_path": str(model_path),
    }
