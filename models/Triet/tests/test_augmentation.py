"""Augmentation consistency and train-split isolation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_extraction.geometric_features import extract_geometric_features
from src.preprocessing.coordinate_normalizer import normalize_landmarks
from src.training.augmentation import augment_landmark_samples
import src.training.trainer as trainer_module


def make_synthetic_91d_samples(count: int = 4) -> np.ndarray:
    rng = np.random.default_rng(123)
    samples = rng.normal(0.0, 0.08, size=(count, 21, 3)).astype(np.float32)
    samples[:, 0, :] = 0.0
    samples = np.asarray(
        [normalize_landmarks(sample).reshape(63) for sample in samples],
        dtype=np.float32,
    )

    if count:
        output = []
        for row in samples:
            normalized = row.reshape(21, 3)
            output.append(extract_geometric_features(normalized, baseline_63=False))
        return np.asarray(output, dtype=np.float32)
    return np.empty((0, 91), dtype=np.float32)


def test_augmentation_does_not_mutate_source_and_recomputes_91d() -> None:
    x = make_synthetic_91d_samples(6)
    y = np.arange(6, dtype=np.int32) % 26
    original_x = x.copy()
    original_y = y.copy()

    augmented_x, augmented_y = augment_landmark_samples(
        x,
        y,
        feature_mode="91d",
        random_seed=42,
    )

    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    assert augmented_x.shape == (6, 91)
    assert augmented_y.shape == (6,)
    assert not np.array_equal(augmented_x, x)

    sample = augmented_x[0]
    normalized = normalize_landmarks(sample[:63].reshape(21, 3))
    recomputed = extract_geometric_features(normalized, baseline_63=False)
    np.testing.assert_allclose(
        sample,
        recomputed,
        atol=1e-5,
        rtol=0.0,
    )


def test_trainer_augments_only_train(monkeypatch, tmp_path: Path) -> None:
    train_x = make_synthetic_91d_samples(8)
    val_x = make_synthetic_91d_samples(4)
    train_frame = pd.DataFrame(train_x)
    train_frame["label"] = [f"{chr(65 + index % 26)}" for index in range(8)]
    val_frame = pd.DataFrame(val_x)
    val_frame["label"] = [f"{chr(65 + index % 26)}" for index in range(4)]

    train_frame.columns = [
        *[f"feature_{index}" for index in range(91)],
        "label",
    ]
    val_frame.columns = [
        *[f"feature_{index}" for index in range(91)],
        "label",
    ]

    train_frame_original = train_frame.copy()
    val_frame_original = val_frame.copy()
    augmented_inputs: list[np.ndarray] = []

    def fake_augment(X, y, feature_mode, random_seed):
        augmented_inputs.append(np.asarray(X).copy())
        return np.asarray(X, dtype=np.float32).copy(), np.asarray(y).copy()

    monkeypatch.setattr(trainer_module, "augment_landmark_samples", fake_augment)
    monkeypatch.setattr(
        trainer_module,
        "export_onnx",
        lambda keras_path, onnx_path: onnx_path,
    )

    labels = {chr(65 + index): index for index in range(26)}
    model, _ = trainer_module.train_keras_model(
        experiment_name="exp3_mlp_opt",
        train_frame=train_frame,
        val_frame=val_frame,
        label_to_index=labels,
        output_dir=tmp_path,
        checkpoint_dir=tmp_path,
        history_path=tmp_path / "history.json",
        seed=42,
        batch_size=4,
        epochs=1,
        augment_train=True,
        label_smoothing=0.05,
    )

    pd.testing.assert_frame_equal(train_frame, train_frame_original)
    pd.testing.assert_frame_equal(val_frame, val_frame_original)
    assert len(augmented_inputs) == 1
    assert augmented_inputs[0].shape == (8, 91)
    model.stop_training = True
