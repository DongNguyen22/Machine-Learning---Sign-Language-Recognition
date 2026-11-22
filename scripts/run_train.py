"""Train ASL comparison experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.trainer import (
    load_feature_frame,
    train_keras_model,
    train_traditional_model,
)
from src.utils.config_loader import load_config


def _read_label_map(config: object) -> dict[str, int]:
    return dict(config.label_to_index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL comparison experiments")
    parser.add_argument(
        "--exp",
        choices=("exp1", "exp2", "exp3"),
        required=True,
        help="Experiment group to run",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--augment-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment training data only (enabled by default)",
    )
    args = parser.parse_args()

    config = load_config()
    processed = ROOT / "datasets" / "processed"
    train_path = processed / "train_features.parquet"
    val_path = processed / "val_features.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit("Thiếu train_features.parquet hoặc val_features.parquet.")

    train_frame = load_feature_frame(train_path)
    val_frame = load_feature_frame(val_path)
    label_to_index = _read_label_map(config)

    checkpoint_dir = ROOT / "models" / "checkpoints"
    production_dir = ROOT / "models" / "production"
    reports_dir = ROOT / "reports" / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    experiment_names = {
        "exp1": ("exp1_knn", "exp1_mlp_base"),
        "exp2": ("exp2_mlp_geo",),
        "exp3": ("exp3_mlp_opt", "exp3_svm_opt"),
    }[args.exp]

    all_metrics: dict[str, object] = {}
    for experiment_name in experiment_names:
        if experiment_name.endswith("_mlp_base") or experiment_name.endswith("_mlp_geo") or experiment_name.endswith("_mlp_opt"):
            input_columns = [
                column
                for column in train_frame.columns
                if column.startswith("feature_")
            ]
            if experiment_name == "exp1_mlp_base" and len(input_columns) != 63:
                raise SystemExit(
                    "exp1_mlp_base yêu cầu Parquet 63 chiều; "
                    "chạy run_build_dataset.py --baseline-63."
                )
            if experiment_name != "exp1_mlp_base" and len(input_columns) != 91:
                raise SystemExit(
                    f"{experiment_name} yêu cầu Parquet 91 chiều."
                )

            _, metrics = train_keras_model(
                experiment_name=experiment_name,
                train_frame=train_frame,
                val_frame=val_frame,
                label_to_index=label_to_index,
                output_dir=production_dir,
                checkpoint_dir=checkpoint_dir,
                history_path=reports_dir / f"{experiment_name}_history.json",
                seed=config.split.random_seed,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                augment_train=args.augment_train,
                label_smoothing=(
                    args.label_smoothing
                    if experiment_name == "exp3_mlp_opt"
                    else 0.0
                ),
                train_dataset_path=train_path,
                val_dataset_path=val_path,
            )
        else:
            _, metrics = train_traditional_model(
                experiment_name=experiment_name,
                train_frame=train_frame,
                val_frame=val_frame,
                label_to_index=label_to_index,
                output_dir=production_dir,
                seed=config.split.random_seed,
                train_dataset_path=train_path,
                val_dataset_path=val_path,
            )
        all_metrics[experiment_name] = metrics

    metrics_path = reports_dir / f"{args.exp}_metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
