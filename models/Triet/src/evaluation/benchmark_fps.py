"""Model file size and CPU inference benchmark."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def model_size(model_path: Path | str) -> dict[str, float]:
    path = Path(model_path)
    bytes_count = path.stat().st_size if path.is_file() else 0
    return {
        "bytes": float(bytes_count),
        "kb": bytes_count / 1024.0,
        "mb": bytes_count / (1024.0 * 1024.0),
    }


def benchmark_model(
    model: Any,
    sample: np.ndarray,
    warmup: int = 50,
    iterations: int = 1000,
) -> dict[str, float]:
    x = np.asarray(sample, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if not np.isfinite(x).all():
        raise ValueError("Sample suy luận chứa NaN hoặc Inf.")

    def predict_once() -> None:
        if hasattr(model, "predict"):
            if hasattr(model, "predict_proba"):
                model.predict_proba(x)
            else:
                model.predict(x, verbose=0)
        elif hasattr(model, "run"):
            model.run(None, {model.get_inputs()[0].name: x})
        else:
            raise TypeError("Mô hình không hỗ trợ suy luận predict/predict_proba/run.")

    for _ in range(warmup):
        predict_once()

    start = time.perf_counter()
    for _ in range(iterations):
        predict_once()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / iterations
    fps = 1000.0 / latency_ms if latency_ms else 0.0
    return {
        "warmup": float(warmup),
        "iterations": float(iterations),
        "latency_ms": float(latency_ms),
        "fps": float(fps),
        "elapsed_seconds": float(elapsed),
    }
