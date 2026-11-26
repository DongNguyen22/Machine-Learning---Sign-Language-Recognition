"""Keras, ONNX and sklearn pipeline serialization tests."""

from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort

from src.models.mlp_classifier import (
    build_optimized_mlp,
    prepare_targets,
)
from src.training.trainer import export_onnx


def make_synthetic_model_data(
    count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 0.5, size=(count, 91)).astype(np.float32)
    y = np.asarray([index % 26 for index in range(count)], dtype=np.int32)
    return x, y


def test_keras_label_smoothing_and_onnx_serialization(tmp_path: Path) -> None:
    x, y = make_synthetic_model_data()
    y_one_hot = prepare_targets(y, num_classes=26, label_smoothing=0.05)

    model = build_optimized_mlp(
        input_dim=91,
        num_classes=26,
        learning_rate=1e-3,
        label_smoothing=0.05,
        x_train=x,
    )
    model.fit(
        x,
        y_one_hot,
        epochs=2,
        batch_size=32,
        verbose=0,
    )

    keras_path = tmp_path / "test_mlp.keras"
    model.save(keras_path)
    assert keras_path.is_file()

    from tensorflow import keras
    restored = keras.models.load_model(keras_path)
    keras_predictions = restored.predict(x[:20], verbose=0)

    onnx_path = tmp_path / "test_mlp.onnx"
    export_onnx(keras_path, onnx_path)
    assert onnx_path.is_file()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x[:20].astype(np.float32)})
    onnx_predictions = np.asarray(output[0])

    np.testing.assert_allclose(
        keras_predictions,
        onnx_predictions,
        atol=1e-4,
        rtol=0.0,
    )


def test_sklearn_pipeline_joblib_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    x = rng.normal(0.0, 1.0, size=(100, 91)).astype(np.float32)
    y = np.asarray([index % 26 for index in range(100)], dtype=np.int32)

    pipeline = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ])
    pipeline.fit(x, y)
    before = pipeline.predict(x[:30])

    model_path = tmp_path / "test_knn.joblib"
    joblib.dump(pipeline, model_path)
    restored = joblib.load(model_path)
    after = restored.predict(x[:30])

    np.testing.assert_array_equal(before, after)
