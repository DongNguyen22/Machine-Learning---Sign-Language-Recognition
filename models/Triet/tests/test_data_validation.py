"""Validation tests for manifest and extracted Parquet datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "datasets" / "processed"
MANIFEST_PATH = ROOT / "datasets" / "metadata" / "split_manifest.csv"
ALL_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PARQUET_NAMES = (
    "train_features.parquet",
    "val_features.parquet",
    "test_in_domain_features.parquet",
    "test_webcam_features.parquet",
)


def test_split_manifest_has_no_cross_split_image_overlap() -> None:
    assert MANIFEST_PATH.is_file(), f"Thiếu manifest: {MANIFEST_PATH}"
    manifest = pd.read_csv(MANIFEST_PATH)
    required = {"image_path", "label", "source", "split"}
    assert required.issubset(manifest.columns)

    split_paths = {
        split: set(
            manifest.loc[
                manifest["split"].eq(split),
                "image_path",
            ].astype(str)
        )
        for split in ("train", "val", "test_in_domain")
    }

    assert split_paths["train"].isdisjoint(split_paths["val"])
    assert split_paths["train"].isdisjoint(split_paths["test_in_domain"])
    assert split_paths["val"].isdisjoint(split_paths["test_in_domain"])


@pytest.mark.parametrize("filename", PARQUET_NAMES)
def test_parquet_contains_no_nan_or_inf(filename: str) -> None:
    parquet_path = PROCESSED_DIR / filename
    if not parquet_path.is_file():
        pytest.skip(f"Chưa có dữ liệu thực: {parquet_path}")
    frame = pd.read_parquet(parquet_path)

    numeric = frame.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    values = numeric.to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()


def test_in_domain_and_webcam_parquets_cover_all_labels() -> None:
    frames = []
    for filename in PARQUET_NAMES:
        parquet_path = PROCESSED_DIR / filename
        if not parquet_path.is_file():
            continue
        frame = pd.read_parquet(parquet_path)
        if "label" in frame.columns:
            frames.append(frame[["label"]])

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined.empty:
        pytest.skip("Chưa có dữ liệu trong datasets/processed; bỏ qua kiểm tra 26 nhãn.")

    labels = set(combined["label"].dropna().astype(str))
    missing = ALL_LABELS - labels
    assert not missing, f"Thiếu nhãn lớp: {sorted(missing)}"
