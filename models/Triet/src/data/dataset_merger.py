"""Scan and merge the two ASL image datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class DatasetRecord:
    image_path: Path
    label: str
    source: str
    sha256_hash: str = ""


class RawDataPreflightError(RuntimeError):
    """Raised when no real in-domain images are available."""


def normalize_label(folder_name: str) -> str | None:
    """Normalize an image-folder name to one of A-Z."""
    token = re.sub(r"[\s_-]+", "", folder_name).upper()
    if token in VALID_LABELS:
        return token

    match = re.fullmatch(r"(?:LETTER|ASL|CLASS|SIGN)?([A-Z])(?:LETTER|ASL|CLASS|SIGN)?", token)
    if match:
        return match.group(1)
    return None


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_raw_data(
    kaggle_dir: Path | str,
    mendeley_dir: Path | str,
) -> dict[str, int]:
    kaggle = Path(kaggle_dir)
    mendeley = Path(mendeley_dir)
    kaggle_count = sum(
        1 for path in kaggle.rglob("*") if is_image_file(path)
    ) if kaggle.exists() else 0
    mendeley_count = sum(
        1 for path in mendeley.rglob("*") if is_image_file(path)
    ) if mendeley.exists() else 0
    total = kaggle_count + mendeley_count
    if total == 0:
        raise RawDataPreflightError(
            "PREFLIGHT FAILED: No real image files found in "
            f"{kaggle.resolve()} hoặc {mendeley.resolve()}.\n"
            "Place .jpg/.jpeg/.png files into A-Z label subfolders, for example:\n"
            f"  {kaggle.resolve()}\\A\\anh.jpg\n"
            f"  {mendeley.resolve()}\\A\\anh.jpg\n"
            "Sau đó chạy lại scripts/run_create_manifest.py."
        )
    return {"kaggle": kaggle_count, "mendeley": mendeley_count, "total": total}


def scan_dataset(dataset_dir: Path | str, source: str) -> list[DatasetRecord]:
    """Recursively scan a dataset and retain only A-Z folder labels."""
    root = Path(dataset_dir)
    if not root.exists():
        return []

    records: list[DatasetRecord] = []
    for path in sorted(root.rglob("*")):
        if not is_image_file(path):
            continue
        label = normalize_label(path.parent.name)
        if label is not None:
            records.append(DatasetRecord(
                image_path=path.resolve(),
                label=label,
                source=source,
                sha256_hash=sha256_file(path),
            ))
    return records


def merge_datasets(
    kaggle_dir: Path | str,
    mendeley_dir: Path | str,
) -> list[DatasetRecord]:
    """Merge Kaggle and Mendeley records under common A-Z labels."""
    return [
        *scan_dataset(kaggle_dir, "kaggle"),
        *scan_dataset(mendeley_dir, "mendeley"),
    ]


def deduplicate_records(
    records: Iterable[DatasetRecord],
) -> tuple[list[DatasetRecord], dict[str, int]]:
    """Keep the first byte-identical image and count removed copies per source."""
    unique: list[DatasetRecord] = []
    seen_hashes: set[str] = set()
    removed_by_source = {"kaggle": 0, "mendeley": 0, "webcam": 0, "total": 0}

    for record in records:
        digest = record.sha256_hash or sha256_file(record.image_path)
        if digest in seen_hashes:
            if record.source in removed_by_source:
                removed_by_source[record.source] += 1
            removed_by_source["total"] += 1
            continue
        seen_hashes.add(digest)
        unique.append(DatasetRecord(
            image_path=record.image_path,
            label=record.label,
            source=record.source,
            sha256_hash=digest,
        ))
    return unique, removed_by_source


def scan_all_datasets(
    kaggle_dir: Path | str,
    mendeley_dir: Path | str,
    webcam_dir: Path | str,
) -> list[DatasetRecord]:
    records = merge_datasets(kaggle_dir, mendeley_dir)
    records.extend(scan_dataset(webcam_dir, "webcam"))
    return records


def count_by_label(records: Iterable[DatasetRecord]) -> Mapping[str, int]:
    counts = {label: 0 for label in VALID_LABELS}
    for record in records:
        if record.label in counts:
            counts[record.label] += 1
    return counts

