"""Stratified splitting for ASL image records."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from src.data.dataset_merger import DatasetRecord, deduplicate_records, sha256_file

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_records(
    records: Sequence[DatasetRecord],
    seed: int = 42,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict[str, str]:
    """Assign a train/val/test_in_domain split by stratifying on label."""
    if abs((train_ratio + val_ratio + (1.0 - train_ratio - val_ratio)) - 1.0) > 1e-6:
        raise ValueError("Tổng train_ratio, val_ratio và test_ratio phải bằng 1.")

    grouped: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        n = len(items)

        if n == 0:
            continue
        if n == 1:
            train_count = 1
            val_count = 0
        elif n == 2:
            train_count = 1
            val_count = 0
        elif n == 3:
            train_count = 1
            val_count = 1
        else:
            train_count = max(1, int(round(n * train_ratio)))
            val_count = max(1, int(round(n * val_ratio)))
            test_count = n - train_count - val_count
            if test_count < 1:
                train_count = max(1, n - val_count - 1)

        test_count = n - train_count - val_count
        for record in items[:train_count]:
            assignments[str(record.image_path)] = "train"
        for record in items[train_count:train_count + val_count]:
            assignments[str(record.image_path)] = "val"
        for record in items[train_count + val_count:train_count + val_count + test_count]:
            assignments[str(record.image_path)] = "test_in_domain"
    return assignments


def write_split_manifest(
    records: Sequence[DatasetRecord],
    output_path: Path | str,
    seed: int = 42,
    stats_path: Path | str | None = None,
) -> Path:
    """Ghi manifest train/val/test in-domain và webcam sau khi deduplicate."""
    in_domain = [record for record in records if record.source != "webcam"]
    webcam = [record for record in records if record.source == "webcam"]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    in_domain, in_domain_removed = deduplicate_records(in_domain)
    webcam, webcam_removed = deduplicate_records(webcam)

    # Một ảnh webcam trùng dữ liệu in-domain không được làm tập ngoài miền giả lập.
    in_domain_hashes = {
        record.sha256_hash or sha256_file(record.image_path)
        for record in in_domain
    }
    filtered_webcam: list[DatasetRecord] = []
    webcam_cross_source_removed = 0
    for record in webcam:
        digest = record.sha256_hash or sha256_file(record.image_path)
        if digest in in_domain_hashes:
            webcam_cross_source_removed += 1
            webcam_removed["total"] += 1
            continue
        filtered_webcam.append(record)
    webcam = filtered_webcam
    assignments = split_records(in_domain, seed=seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "label", "source", "split", "sha256_hash"],
        )
        writer.writeheader()
        for record in in_domain:
            writer.writerow({
                "image_path": str(record.image_path),
                "label": record.label,
                "source": record.source,
                "split": assignments[str(record.image_path)],
                "sha256_hash": record.sha256_hash or sha256_file(record.image_path),
            })
        for record in webcam:
            writer.writerow({
                "image_path": str(record.image_path),
                "label": record.label,
                "source": record.source,
                "split": "test_webcam",
                "sha256_hash": record.sha256_hash or sha256_file(record.image_path),
            })

    if stats_path is not None:
        stats_file = Path(stats_path)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, object] = {}
        if stats_file.is_file():
            try:
                existing = json.loads(stats_file.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except json.JSONDecodeError:
                existing = {}
        existing["deduplication"] = {
            "removed_total": (
                in_domain_removed["total"]
                + webcam_removed["total"]
            ),
            "removed_by_source": {
                "kaggle": in_domain_removed["kaggle"],
                "mendeley": in_domain_removed["mendeley"],
                "webcam": (
                    webcam_removed["webcam"] + webcam_cross_source_removed
                ),
            },
            "cross_source_webcam_removed": webcam_cross_source_removed,
            "kept_in_domain": len(in_domain),
            "kept_webcam": len(webcam),
        }
        stats_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return path
