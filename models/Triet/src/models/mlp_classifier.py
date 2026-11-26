"""Keras MLP classifiers for ASL feature vectors."""

from __future__ import annotations

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def sparse_crossentropy(label_smoothing: float = 0.0) -> keras.losses.Loss:
    if label_smoothing != 0.0:
        raise ValueError(
            "SparseCategoricalCrossentropy trong Keras 3 không hỗ trợ label_smoothing; "
            "dùng categorical_crossentropy cho chế độ này."
        )
    return keras.losses.SparseCategoricalCrossentropy(
        name="loss",
    )


def categorical_crossentropy(label_smoothing: float = 0.05) -> keras.losses.Loss:
    return keras.losses.CategoricalCrossentropy(
        name="loss",
        label_smoothing=label_smoothing,
    )


def prepare_targets(
    y: np.ndarray,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> np.ndarray:
    integer_targets = np.asarray(y, dtype=np.int32).reshape(-1)
    if label_smoothing > 0:
        return keras.utils.to_categorical(
            integer_targets,
            num_classes=num_classes,
        ).astype(np.float32)
    return integer_targets


def build_baseline_mlp(
    input_dim: int = 63,
    num_classes: int = 26,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0,
    x_train: np.ndarray | None = None,
) -> keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    if x_train is not None:
        normalization = layers.Normalization(name="train_normalization")
        normalization.adapt(np.asarray(x_train, dtype=np.float32))
        x = normalization(x)
    x = layers.Dense(64, activation="relu", name="hidden")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="mlp_baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=(
            categorical_crossentropy(label_smoothing)
            if label_smoothing > 0
            else sparse_crossentropy(0.0)
        ),
        metrics=["accuracy"],
    )
    return model


def build_optimized_mlp(
    input_dim: int = 91,
    num_classes: int = 26,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0,
    x_train: np.ndarray | None = None,
) -> keras.Model:
    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    if x_train is not None:
        normalization = layers.Normalization(name="train_normalization")
        normalization.adapt(np.asarray(x_train, dtype=np.float32))
        x = normalization(x)
    x = layers.Dense(256, name="dense_256")(x)
    x = layers.BatchNormalization(name="batch_norm_256")(x)
    x = layers.Activation("relu", name="relu_256")(x)
    x = layers.Dropout(0.3, name="dropout_256")(x)

    x = layers.Dense(128, name="dense_128")(x)
    x = layers.BatchNormalization(name="batch_norm_128")(x)
    x = layers.Activation("relu", name="relu_128")(x)
    x = layers.Dropout(0.2, name="dropout_128")(x)

    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="mlp_optimized")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=(
            categorical_crossentropy(label_smoothing)
            if label_smoothing > 0
            else sparse_crossentropy(0.0)
        ),
        metrics=["accuracy"],
    )
    return model


build_improved_mlp = build_optimized_mlp
