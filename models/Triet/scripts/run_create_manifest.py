"""Create the stratified dataset manifest from raw Kaggle/Mendeley/webcam folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_splitter import write_split_manifest
from src.data.dataset_merger import (
    RawDataPreflightError,
    preflight_raw_data,
    scan_all_datasets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo manifest phân chia ASL")
    parser.add_argument("--kaggle", type=Path, default=ROOT / "datasets/raw/kaggle_asl")
    parser.add_argument("--mendeley", type=Path, default=ROOT / "datasets/raw/mendeley_asl")
    parser.add_argument("--webcam", type=Path, default=ROOT / "datasets/raw/webcam_real_test")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/metadata/split_manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        counts = preflight_raw_data(args.kaggle, args.mendeley)
    except RawDataPreflightError as exc:
        raise SystemExit(str(exc)) from exc

    records = scan_all_datasets(args.kaggle, args.mendeley, args.webcam)
    in_domain = [
        record for record in records
        if record.source in ("kaggle", "mendeley")
    ]
    if not in_domain:
        raise SystemExit(
            "PREFLIGHT FAILED: found image files, but none are in valid A-Z "
            "label folders.\n"
            "Dat du lieu trong cac thu muc con A, B, ..., Z."
        )
    stats_path = ROOT / "datasets" / "metadata" / "extraction_stats.json"
    path = write_split_manifest(
        records,
        args.output,
        seed=args.seed,
        stats_path=stats_path,
    )
    print(f"Manifest: {path}")
    print(
        "Raw in-domain images "
        f"(Kaggle={counts['kaggle']}, Mendeley={counts['mendeley']}): "
        f"{counts['total']}"
    )


if __name__ == "__main__":
    main()
