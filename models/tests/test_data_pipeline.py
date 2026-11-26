"""Tests for ASL data splitting and geometric feature modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.data_splitter import split_records, write_split_manifest
from src.data.dataset_merger import (
    DatasetRecord,
    deduplicate_records,
    normalize_label,
    preflight_raw_data,
    RawDataPreflightError,
)
from src.feature_extraction.geometric_features import extract_geometric_features
from src.preprocessing.coordinate_normalizer import normalize_landmarks


def test_normalize_landmarks_centers_and_scales() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[4] = (4.0, 0.0, 0.0)
    normalized = normalize_landmarks(landmarks)

    assert normalized.shape == (21, 3)
    assert np.linalg.norm(normalized[4]) == pytest.approx(1.0)
    assert np.linalg.norm(normalized[0]) == pytest.approx(0.0)
    assert np.isfinite(normalized).all()


def test_normalize_rejects_nan() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[1, 0] = np.nan
    with pytest.raises(ValueError):
        normalize_landmarks(landmarks)


def test_geometric_feature_shapes_and_finite_output() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.0, 1.0, 21)
    normalized = normalize_landmarks(landmarks)

    full = extract_geometric_features(normalized)
    baseline = extract_geometric_features(normalized, baseline_63=True)

    assert full.shape == (91,)
    assert full.dtype == np.float32
    assert baseline.shape == (63,)
    assert baseline.dtype == np.float32
    assert np.isfinite(full).all()
    assert np.isfinite(baseline).all()


def test_zero_landmarks_are_finite() -> None:
    normalized = normalize_landmarks(np.zeros((21, 3), dtype=np.float32))
    features = extract_geometric_features(normalized)
    assert features.shape == (91,)
    assert np.isfinite(features).all()


def test_split_records_are_stratified(tmp_path: Path) -> None:
    records = [
        DatasetRecord(tmp_path / f"{letter}_{index}.png", letter, "kaggle")
        for letter in "ABC"
        for index in range(10)
    ]
    split = split_records(records, seed=42)

    assert set(split.values()) == {"train", "val", "test_in_domain"}
    counts = {name: 0 for name in split.values()}
    for value in split.values():
        counts[value] += 1
    assert counts["train"] == 21
    assert counts["val"] == 6
    assert counts["test_in_domain"] == 3


def test_manifest_includes_webcam_split(tmp_path: Path) -> None:
    kaggle = tmp_path / "kaggle"
    kaggle.mkdir()
    for index in range(10):
        (kaggle / f"A_{index}.png").write_bytes(f"in-domain-{index}".encode())
    (kaggle / "A_webcam.png").write_bytes(b"webcam")
    records = [
        DatasetRecord(kaggle / f"A_{index}.png", "A", "kaggle")
        for index in range(10)
    ]
    records.append(DatasetRecord(kaggle / "A_webcam.png", "A", "webcam"))
    output = tmp_path / "metadata" / "split_manifest.csv"

    write_split_manifest(records, output)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert "test_webcam" in lines[-1]


def test_label_normalization() -> None:
    assert normalize_label("A") == "A"
    assert normalize_label("letter_b") == "B"
    assert normalize_label("nothing") is None


def test_deduplicate_records_removes_same_hash(tmp_path: Path) -> None:
    first = tmp_path / "A1.jpg"
    second = tmp_path / "A2.jpg"
    first.write_bytes(b"same-image-content")
    second.write_bytes(b"same-image-content")

    records = [
        DatasetRecord(first, "A", "kaggle", "duplicate-hash"),
        DatasetRecord(second, "A", "mendeley", "duplicate-hash"),
    ]
    unique, removed = deduplicate_records(records)

    assert len(unique) == 1
    assert removed["total"] == 1
    assert removed["mendeley"] == 1


def test_preflight_rejects_empty_raw_folders(tmp_path: Path) -> None:
    kaggle = tmp_path / "kaggle"
    mendeley = tmp_path / "mendeley"
    kaggle.mkdir()
    mendeley.mkdir()

    with pytest.raises(RawDataPreflightError, match="PREFLIGHT FAILED"):
        preflight_raw_data(kaggle, mendeley)
