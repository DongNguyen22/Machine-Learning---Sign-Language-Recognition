"""Kiểm tra nạp cấu hình và hợp đồng 91 đặc trưng."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config_loader import (
    ConfigError,
    load_config,
    project_root_from,
)

ROOT = project_root_from(Path(__file__))


def test_load_project_config_matches_contract() -> None:
    config = load_config()

    assert config.project_name == "asl-alphabet-recognition"
    assert config.paths.root == ROOT
    assert config.paths.raw_kaggle_asl == (ROOT / "datasets/raw/kaggle_asl").resolve()
    assert config.paths.features_parquet.suffix == ".parquet"
    assert config.paths.checkpoints_dir == (ROOT / "models/checkpoints").resolve()
    assert config.paths.metrics_dir == (ROOT / "reports/metrics").resolve()

    assert config.mediapipe.model_complexity == 0
    assert config.mediapipe.min_detection_confidence == pytest.approx(0.7)
    assert config.mediapipe.min_tracking_confidence == pytest.approx(0.6)
    assert config.mediapipe.max_num_hands == 1

    assert config.split.train_ratio == pytest.approx(0.70)
    assert config.split.val_ratio == pytest.approx(0.15)
    assert config.split.test_ratio == pytest.approx(0.15)
    assert config.split.random_seed == 42

    assert config.mlp.input_dim == 91
    assert config.mlp.num_classes == 26
    assert config.mlp.batch_size == 64
    assert config.mlp.epochs == 100
    assert config.mlp.learning_rate == pytest.approx(0.001)
    assert config.mlp.early_stopping_patience == 12
    assert config.mlp.lr_patience == 5

    assert config.postprocess.ema_alpha == pytest.approx(0.4)
    assert config.postprocess.conf_threshold == pytest.approx(0.75)
    assert config.postprocess.margin_threshold == pytest.approx(0.20)
    assert config.postprocess.window_size == 10
    assert config.postprocess.min_votes == 7
    assert config.postprocess.hold_seconds == pytest.approx(1.2)

    assert config.index_to_label[0] == "A"
    assert config.index_to_label[25] == "Z"
    assert config.label_to_index["M"] == 12
    assert len(config.index_to_label) == 26


def test_feature_schema_covers_91_dimensions() -> None:
    schema = load_config().features

    assert schema.input_dim == 91
    assert [group.size for group in schema.groups] == [63, 15, 10, 3]
    assert schema.groups[0].start == 0
    assert schema.groups[-1].end == 91
    assert len(schema.landmarks) == 21
    assert len(schema.bones) == 0
    assert len(schema.angles) == 0
    assert schema.landmark_feature_index(0, 0) == 0
    assert schema.landmark_feature_index(20, 2) == 62
    with pytest.raises(ValueError):
        schema.landmark_feature_index(-1, 0)


def test_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    broken = source.replace("train_ratio: 0.70", "train_ratio: 0.50", 1)
    target = config_dir / "config.yaml"
    target.write_text(broken, encoding="utf-8")

    with pytest.raises(ConfigError, match="bằng 1"):
        load_config(target)


def test_missing_label_map_raises_file_not_found(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    source = (ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
    broken = source.replace(
        "label_map: configs/label_map.json",
        "label_map: configs/missing_label_map.json",
        1,
    )
    target = config_dir / "config.yaml"
    target.write_text(broken, encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing_label_map"):
        load_config(target)
