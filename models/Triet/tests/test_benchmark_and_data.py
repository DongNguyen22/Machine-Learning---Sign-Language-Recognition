"""FPS batch-size and SHA-256 deduplication tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.dataset_merger import DatasetRecord, deduplicate_records
from src.evaluation.benchmark_fps import benchmark_model


class BatchTrackingModel:
    def __init__(self) -> None:
        self.last_batch_size = 0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self.last_batch_size = x.shape[0]
        return np.ones((x.shape[0], 26), dtype=np.float32) / 26.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self.last_batch_size = x.shape[0]
        return np.zeros(x.shape[0], dtype=np.int32)


def test_benchmark_batch_size_one() -> None:
    model = BatchTrackingModel()
    sample = np.zeros((1, 91), dtype=np.float32)

    result = benchmark_model(model, sample, warmup=2, iterations=5)

    assert model.last_batch_size == 1
    assert result["latency_ms"] >= 0.0
    assert result["fps"] >= 0.0


def test_benchmark_large_batch_256() -> None:
    model = BatchTrackingModel()
    sample = np.zeros((256, 91), dtype=np.float32)

    result = benchmark_model(model, sample, warmup=2, iterations=5)

    assert model.last_batch_size == 256
    assert result["latency_ms"] >= 0.0
    assert result["fps"] >= 0.0


def test_sha256_deduplication_same_content_different_names(tmp_path: Path) -> None:
    first = tmp_path / "A_one.jpg"
    second = tmp_path / "A_copy.png"
    first.write_bytes(b"identical-image-byte-content")
    second.write_bytes(b"identical-image-byte-content")

    records = [
        DatasetRecord(first, "A", "kaggle"),
        DatasetRecord(second, "A", "mendeley"),
    ]
    unique, removed = deduplicate_records(records)

    assert len(unique) == 1
    assert removed["total"] == 1
    assert removed["mendeley"] == 1
