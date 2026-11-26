"""Tests for model factory, callbacks, augmentation, and history utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from src.models.model_factory import create_model
from src.models.mlp_classifier import prepare_targets
from src.training.augmentation import augment_landmark_samples, augment_train_coordinates
from src.training.callbacks import create_training_callbacks


@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("exp1_knn", KNeighborsClassifier),
        ("exp3_svm_opt", SVC),
    ],
)
def test_traditional_model_factory(name: str, expected_kind: type) -> None:
    model = create_model(name, input_dim=91)
    assert isinstance(model, Pipeline)
    if name == "exp3_svm_opt":
        calibrated = model.named_steps["svm"]
        assert isinstance(calibrated, CalibratedClassifierCV)
        assert isinstance(calibrated.estimator, SVC)
        assert calibrated.cv == 3
    else:
        assert any(isinstance(step, expected_kind) for step in model.named_steps.values())


def test_mlp_model_factory_shapes() -> None:
    baseline = create_model("exp1_mlp_base", input_dim=63)
    optimized = create_model("exp3_mlp_opt", input_dim=91)

    assert baseline.input_shape == (None, 63)
    assert baseline.output_shape == (None, 26)
    assert optimized.input_shape == (None, 91)
    assert optimized.output_shape == (None, 26)


def test_label_smoothing_target_is_one_hot() -> None:
    integer_targets = np.asarray([0, 25, 12], dtype=np.int32)

    sparse = prepare_targets(integer_targets, 26, label_smoothing=0.0)
    one_hot = prepare_targets(integer_targets, 26, label_smoothing=0.05)

    assert sparse.shape == (3,)
    assert sparse.dtype == np.int32
    assert one_hot.shape == (3, 26)
    assert one_hot.dtype == np.float32
    assert np.isfinite(one_hot).all()


def test_augmentation_changes_only_a_copy_of_train_data() -> None:
    train = np.zeros((8, 91), dtype=np.float32)
    train[:, :63] = np.arange(63, dtype=np.float32) / 100.0
    train_original = train.copy()
    val = train.copy()
    val_original = val.copy()

    augmented = augment_train_coordinates(train, seed=42)
    np.testing.assert_array_equal(train, train_original)
    np.testing.assert_array_equal(val, val_original)

    assert augmented.shape == train.shape
    assert augmented.dtype == np.float32
    assert np.isfinite(augmented).all()
    assert not np.array_equal(augmented, train)


def test_augmented_91d_vector_is_recomputed_from_landmarks() -> None:
    train = np.zeros((4, 91), dtype=np.float32)
    train[:, :63] = np.arange(63, dtype=np.float32) / 100.0
    y = np.arange(4, dtype=np.int32)

    augmented, augmented_y = augment_landmark_samples(
        train,
        y,
        feature_mode="91d",
        random_seed=42,
    )

    assert augmented.shape == (4, 91)
    assert augmented.dtype == np.float32
    assert np.array_equal(augmented_y, y)
    assert np.isfinite(augmented).all()


def test_callbacks_are_configured(tmp_path: Path) -> None:
    callbacks = create_training_callbacks(tmp_path / "model.keras")
    names = {callback.__class__.__name__ for callback in callbacks}

    assert "EarlyStopping" in names
    assert "ReduceLROnPlateau" in names
    assert "ModelCheckpoint" in names
