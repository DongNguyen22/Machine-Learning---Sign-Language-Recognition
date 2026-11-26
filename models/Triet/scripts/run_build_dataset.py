"""Build extracted train/validation/test feature Parquet files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.hand_detector import HandDetector

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.feature_extraction.geometric_features import extract_geometric_features
from src.preprocessing.coordinate_normalizer import normalize_landmarks
from src.data.dataset_merger import (
    RawDataPreflightError,
    preflight_raw_data,
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_manifest(
    manifest_path: Path,
    processed_dir: Path,
    stats_path: Path,
    baseline_63: bool = False,
) -> dict[str, dict[str, object]]:
    manifest = _read_manifest(manifest_path)
    split_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    stats: dict[str, dict[str, object]] = {
        label: {"total": 0, "extracted": 0, "failed": 0, "failure_rate": 0.0}
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    }

    detector = HandDetector(static_image_mode=True)
    try:
        for row in tqdm(manifest, desc="Trích xuất đặc trưng", unit="ảnh"):
            label = str(row["label"]).strip().upper()
            image_path = Path(row["image_path"])
            split_name = str(row["split"]).strip()

            stats[label]["total"] += 1
            try:
                import cv2

                image = cv2.imread(str(image_path))
                landmarks = detector.process(image)
                if landmarks is None:
                    stats[label]["failed"] += 1
                    continue
                normalized = normalize_landmarks(landmarks)
                features = extract_geometric_features(
                    normalized,
                    baseline_63=baseline_63,
                )
                if features.shape != ((63,) if baseline_63 else (91,)):
                    raise ValueError("Vector đặc trưng có kích thước không hợp lệ.")
                if not np.isfinite(features).all():
                    raise ValueError("Vector đặc trưng chứa NaN hoặc Inf.")
            except (ValueError, OSError, RuntimeError):
                stats[label]["failed"] += 1
                continue

            result: dict[str, object] = {
                "image_path": str(image_path),
                "label": label,
                "source": str(row.get("source", "")),
                "split": split_name,
            }
            for index, value in enumerate(features):
                result[f"feature_{index}"] = float(value)
            split_rows[split_name].append(result)
            stats[label]["extracted"] += 1
    finally:
        detector.close()

    processed_dir.mkdir(parents=True, exist_ok=True)
    split_outputs = {
        "train": processed_dir / "train_features.parquet",
        "val": processed_dir / "val_features.parquet",
        "test_in_domain": processed_dir / "test_in_domain_features.parquet",
        "test_webcam": processed_dir / "test_webcam_features.parquet",
    }
    for split_name, output_path in split_outputs.items():
        rows = split_rows.get(split_name, [])
        if rows:
            frame = pd.DataFrame(rows)
        else:
            frame = pd.DataFrame(
                columns=["image_path", "label", "source", "split"]
                + [f"feature_{index}" for index in range(63 if baseline_63 else 91)]
            )
        frame.to_parquet(output_path, index=False, engine="pyarrow")

    active_stats: dict[str, dict[str, object]] = {}
    overall_total = 0
    overall_failed = 0
    overall_extracted = 0
    for label, counts in stats.items():
        total = int(counts["total"])
        extracted = int(counts["extracted"])
        failed = int(counts["failed"])
        counts["failure_rate"] = float(failed / total) if total else 0.0
        overall_total += total
        overall_failed += failed
        overall_extracted += extracted
        if total:
            active_stats[label] = counts

    report = {
        "overall": {
            "total": overall_total,
            "extracted": overall_extracted,
            "failed": overall_failed,
            "failure_rate": float(overall_failed / overall_total)
            if overall_total
            else 0.0,
        },
        "by_label": active_stats,
        "feature_count": 63 if baseline_63 else 91,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    existing_report: dict[str, object] = {}
    if stats_path.is_file():
        try:
            existing_report = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(existing_report, dict):
                existing_report = {}
        except json.JSONDecodeError:
            existing_report = {}
    if "deduplication" in existing_report:
        report["deduplication"] = existing_report["deduplication"]
    stats_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Trích xuất đặc trưng ASL")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "datasets" / "metadata" / "split_manifest.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "datasets" / "processed",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "datasets" / "metadata" / "extraction_stats.json",
    )
    parser.add_argument(
        "--baseline-63",
        action="store_true",
        help="Chỉ xuất 63 tọa độ (Experiment 1 Baseline)",
    )
    args = parser.parse_args()

    try:
        preflight_raw_data(
            ROOT / "datasets" / "raw" / "kaggle_asl",
            ROOT / "datasets" / "raw" / "mendeley_asl",
        )
    except RawDataPreflightError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.manifest.is_file():
        raise SystemExit(f"Không tìm thấy manifest: {args.manifest}")
    manifest = _read_manifest(args.manifest)
    if not manifest:
        raise SystemExit(
            "PREFLIGHT FAILED: split_manifest.csv has 0 image rows.\n"
            "No real dataset has been loaded or the manifest was not created.\n"
            "Place .jpg/.jpeg/.png images into Kaggle/Mendeley A-Z folders, "
            "then run python scripts/run_create_manifest.py."
        )
    report = extract_manifest(
        args.manifest,
        args.processed_dir,
        args.stats,
        baseline_63=args.baseline_63,
    )
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
