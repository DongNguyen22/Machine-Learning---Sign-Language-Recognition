"""Nạp cấu hình tập trung của đồ án ASL Alphabet.

Mọi đường dẫn tương đối trong ``configs/config.yaml`` được neo vào thư mục
gốc dự án (thư mục cha của ``configs/``). File thiếu được báo bằng
``FileNotFoundError``; nội dung sai được báo bằng ``ConfigError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_CONFIG_RELATIVE = Path("configs") / "config.yaml"
_RATIO_TOLERANCE = 1e-6
_LANDMARK_COUNT = 21


class ConfigError(ValueError):
    """File cấu hình tồn tại nhưng nội dung không hợp lệ."""


@dataclass(frozen=True)
class MediaPipeConfig:
    """Tham số ``mp.solutions.hands.Hands``."""

    model_complexity: int
    min_detection_confidence: float
    min_tracking_confidence: float
    max_num_hands: int


@dataclass(frozen=True)
class SplitConfig:
    """Tỉ lệ chia train / validation / test và seed tái lập."""

    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int


@dataclass(frozen=True)
class MLPConfig:
    """Siêu tham số MLP (input 91, 26 lớp)."""

    input_dim: int
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    lr_patience: int


@dataclass(frozen=True)
class PostprocessConfig:
    """Hậu xử lý chuỗi xác suất khi suy luận webcam."""

    ema_alpha: float
    conf_threshold: float
    margin_threshold: float
    window_size: int
    min_votes: int
    hold_seconds: float


@dataclass(frozen=True)
class PathConfig:
    """Đường dẫn đã resolve thành ``Path`` tuyệt đối."""

    root: Path
    raw_kaggle_asl: Path
    raw_mendeley_asl: Path
    raw_webcam_real_test: Path
    processed_dir: Path
    features_parquet: Path
    train_parquet: Path
    val_parquet: Path
    test_parquet: Path
    metadata_dir: Path
    checkpoints_dir: Path
    production_dir: Path
    figures_dir: Path
    metrics_dir: Path
    logs_dir: Path
    label_map: Path
    feature_config: Path


@dataclass(frozen=True)
class LandmarkSpec:
    """Một mốc trên khung xương MediaPipe Hands."""

    index: int
    name: str


@dataclass(frozen=True)
class BoneSpec:
    """Một đoạn xương và chỉ số đặc trưng tương ứng."""

    feature_index: int
    start: int
    end: int
    name: str


@dataclass(frozen=True)
class AngleSpec:
    """Góc tại một khớp: đỉnh ``vertex``, hai cánh ``arm_a`` và ``arm_b``."""

    feature_index: int
    name: str
    vertex: int
    arm_a: int
    arm_b: int
    description: str


@dataclass(frozen=True)
class FeatureGroupSpec:
    """Một khối liên tục trong vector 91 chiều (end là exclusive)."""

    name: str
    start: int
    end: int
    size: int
    description: str


@dataclass(frozen=True)
class FeatureSchema:
    """Hợp đồng vector đặc trưng dùng chung bởi rút đặc trưng và MLP."""

    name: str
    version: int
    input_dim: int
    dtype: str
    origin_landmark_index: int
    scale_landmark_index: int
    scale_epsilon: float
    landmarks: tuple[LandmarkSpec, ...]
    groups: tuple[FeatureGroupSpec, ...]
    bones: tuple[BoneSpec, ...]
    angles: tuple[AngleSpec, ...]

    def landmark_feature_index(self, landmark_id: int, axis: int) -> int:
        """Chỉ số đặc trưng của trục ``axis`` (0=x, 1=y, 2=z) cho landmark 0..20.

        Args:
            landmark_id: Chỉ số MediaPipe, từ 0 đến 20.
            axis: 0, 1 hoặc 2.

        Returns:
            Chỉ số trong ``[0, 63)``.

        Raises:
            ValueError:             Khi landmark hoặc trục nằm ngoài hợp đồng.
        """
        if landmark_id < 0 or landmark_id > 20:
            raise ValueError(
                "Khối tọa độ chứa landmark 0..20; "
                f"nhận landmark_id={landmark_id}."
            )
        if axis not in (0, 1, 2):
            raise ValueError(f"axis phải là 0, 1 hoặc 2; nhận {axis}.")
        return landmark_id * 3 + axis


@dataclass(frozen=True)
class ProjectConfig:
    """Ảnh chụp bất biến của toàn bộ cấu hình dự án."""

    project_name: str
    version: str
    description: str
    paths: PathConfig
    mediapipe: MediaPipeConfig
    split: SplitConfig
    mlp: MLPConfig
    postprocess: PostprocessConfig
    index_to_label: Mapping[int, str]
    label_to_index: Mapping[str, int]
    features: FeatureSchema


def project_root_from(start: Path | None = None) -> Path:
    """Tìm thư mục gốc bằng cách đi ngược lên đến khi thấy ``configs/config.yaml``.

    Args:
        start: File hoặc thư mục bắt đầu. Mặc định là thư mục làm việc hiện tại.

    Returns:
        Thư mục gốc dự án.

    Raises:
        FileNotFoundError: Không có ``configs/config.yaml`` trên đường đi ngược.
    """
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    searched: list[Path] = []
    for candidate in (current, *current.parents):
        searched.append(candidate)
        marker = candidate / _CONFIG_RELATIVE
        if marker.is_file():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy configs/config.yaml khi đi ngược từ "
        f"{current}. Đã xét: {', '.join(str(path) for path in searched)}."
    )


def load_config(config_path: Path | str | None = None) -> ProjectConfig:
    """Nạp YAML, ánh xạ nhãn và hợp đồng đặc trưng.

    Args:
        config_path: Đường dẫn ``config.yaml``. ``None`` thì tự tìm từ thư mục
            làm việc, rồi từ vị trí file này.

    Returns:
        ``ProjectConfig`` đã validate.

    Raises:
        FileNotFoundError: YAML, ``label_map.json`` hoặc ``feature_config.json``
            không tồn tại hoặc không đọc được.
        ConfigError: Tỉ lệ chia tập, số chiều hoặc ánh xạ nhãn không khớp.
    """
    path = _resolve_config_path(config_path)
    raw = _read_yaml_mapping(path)
    root = _root_for_config_file(path)
    paths = _parse_paths(root, raw)
    mediapipe = _parse_mediapipe(raw)
    split = _parse_split(raw)
    mlp = _parse_mlp(raw)
    postprocess = _parse_postprocess(raw)
    index_to_label, label_to_index = load_label_map(paths.label_map)
    features = load_feature_schema(paths.feature_config)
    project = _require_mapping(raw, "project", "project")
    config = ProjectConfig(
        project_name=_require_str(project, "name", "project.name"),
        version=_require_str(project, "version", "project.version"),
        description=_require_str(project, "description", "project.description"),
        paths=paths,
        mediapipe=mediapipe,
        split=split,
        mlp=mlp,
        postprocess=postprocess,
        index_to_label=index_to_label,
        label_to_index=label_to_index,
        features=features,
    )
    _validate_cross_refs(config)
    return config


def load_label_map(path: Path | str) -> tuple[Mapping[int, str], Mapping[str, int]]:
    """Đọc ánh xạ lớp.

    Chấp nhận hai dạng:

    * ``{"index_to_label": {"0": "A", ...}}`` (có thể kèm ``label_to_index``)
    * ``{"0": "A", "1": "B", ...}``

    Args:
        path: File JSON.

    Returns:
        Cặp ``(index_to_label, label_to_index)``.

    Raises:
        FileNotFoundError: File không tồn tại hoặc không đọc được.
        ConfigError: Không đủ 26 nhãn A-Z duy nhất.
    """
    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ConfigError(f"label map phải là object JSON: {path}")
    if "index_to_label" in payload:
        raw_map = payload["index_to_label"]
        if not isinstance(raw_map, dict):
            raise ConfigError(f"index_to_label phải là object: {path}")
    else:
        raw_map = payload
    index_to_label = _parse_index_to_label(raw_map, Path(path))
    label_to_index = {label: index for index, label in index_to_label.items()}
    if len(label_to_index) != len(index_to_label):
        raise ConfigError(f"Nhãn bị trùng trong {path}.")
    return index_to_label, label_to_index


def load_feature_schema(path: Path | str) -> FeatureSchema:
    """Đọc và kiểm tra hợp đồng 91 đặc trưng.

    Args:
        path: ``configs/feature_config.json``.

    Returns:
        ``FeatureSchema`` đã kiểm tra phủ kín ``[0, input_dim)``.

    Raises:
        FileNotFoundError: File không tồn tại hoặc không đọc được.
        ConfigError: Kích thước nhóm, xương hoặc góc không khớp ``input_dim``.
    """
    payload = _read_json(Path(path))
    if not isinstance(payload, dict):
        raise ConfigError(f"feature config phải là object JSON: {path}")
    file_path = Path(path)
    input_dim = _require_int(payload, "input_dim", "feature.input_dim")
    landmarks = _parse_landmarks(payload, file_path)
    groups_raw = _require_list(payload, "groups", "feature.groups")
    groups: list[FeatureGroupSpec] = []
    bones: tuple[BoneSpec, ...] = ()
    angles: tuple[AngleSpec, ...] = ()
    for item in groups_raw:
        if not isinstance(item, dict):
            raise ConfigError(f"Mỗi group trong {file_path} phải là object.")
        group = FeatureGroupSpec(
            name=_require_str(item, "name", "feature.groups.name"),
            start=_require_int(item, "start", "feature.groups.start"),
            end=_require_int(item, "end", "feature.groups.end"),
            size=_require_int(item, "size", "feature.groups.size"),
            description=_require_str(item, "description", "feature.groups.description"),
        )
        if group.end - group.start != group.size or group.size <= 0:
            raise ConfigError(
                f"Nhóm {group.name} có size={group.size} nhưng "
                f"khoảng [{group.start}, {group.end}) không khớp."
            )
        groups.append(group)
        if group.name == "bone_lengths":
            bones = _parse_bones(item, group, file_path)
        elif group.name == "joint_angles":
            angles = _parse_angles(item, group, file_path)
    schema_space = _require_mapping(payload, "coordinate_space", "feature.coordinate_space")
    schema = FeatureSchema(
        name=_require_str(payload, "name", "feature.name"),
        version=_require_int(payload, "version", "feature.version"),
        input_dim=input_dim,
        dtype=_require_str(payload, "dtype", "feature.dtype"),
        origin_landmark_index=_require_int(
            schema_space, "origin_landmark_index", "coordinate_space.origin"
        ),
        scale_landmark_index=_require_int(
            schema_space, "scale_landmark_index", "coordinate_space.scale"
        ),
        scale_epsilon=_require_float(
            schema_space, "scale_epsilon", "coordinate_space.scale_epsilon"
        ),
        landmarks=landmarks,
        groups=tuple(groups),
        bones=bones,
        angles=angles,
    )
    _validate_feature_schema(schema, file_path)
    return schema


def ensure_output_directories(config: ProjectConfig) -> None:
    """Tạo thư mục đầu ra mà pipeline được phép ghi.

    Không tạo thư mục dữ liệu thô. Nguồn thiếu phải nổi lên thành
    ``FileNotFoundError`` ở tầng nạp dữ liệu, không được biến thành thư mục rỗng.

    Args:
        config: Cấu hình đã nạp.

    Raises:
        FileNotFoundError: Không tạo được thư mục vì một thành phần cha không
            phải là thư mục.
    """
    output_dirs: Sequence[Path] = (
        config.paths.processed_dir,
        config.paths.metadata_dir,
        config.paths.checkpoints_dir,
        config.paths.production_dir,
        config.paths.figures_dir,
        config.paths.metrics_dir,
        config.paths.logs_dir,
    )
    for directory in output_dirs:
        _mkdir(directory)


def _resolve_config_path(config_path: Path | str | None) -> Path:
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
        return path
    try:
        root = project_root_from(Path.cwd())
    except FileNotFoundError:
        root = project_root_from(Path(__file__))
    found = root / _CONFIG_RELATIVE
    if not found.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {found}")
    return found


def _root_for_config_file(config_file: Path) -> Path:
    resolved = config_file.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return project_root_from(resolved)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML không hợp lệ trong {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Gốc YAML phải là mapping: {path}")
    return data


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON không hợp lệ trong {path}: {exc}") from exc


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise FileNotFoundError(f"Không đọc được file: {path}") from exc


def _mkdir(directory: Path) -> None:
    if directory.exists() and not directory.is_dir():
        raise FileNotFoundError(
            f"Không thể tạo thư mục {directory}: đường dẫn đang là một file."
        )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Không tìm thấy đường dẫn cha để tạo {directory}."
        ) from exc
    except OSError as exc:
        raise FileNotFoundError(f"Không tạo được thư mục {directory}: {exc}") from exc


def _resolve_under_root(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _parse_paths(root: Path, raw: Mapping[str, Any]) -> PathConfig:
    paths = _require_mapping(raw, "paths", "paths")
    data = _require_mapping(paths, "data", "paths.data")
    raw_data = _require_mapping(data, "raw", "paths.data.raw")
    processed = _require_mapping(data, "processed", "paths.data.processed")
    models = _require_mapping(paths, "models", "paths.models")
    reports = _require_mapping(paths, "reports", "paths.reports")
    configs = _require_mapping(paths, "configs", "paths.configs")
    return PathConfig(
        root=root,
        raw_kaggle_asl=_resolve_under_root(
            root, _require_str(raw_data, "kaggle_asl", "paths.data.raw.kaggle_asl")
        ),
        raw_mendeley_asl=_resolve_under_root(
            root, _require_str(raw_data, "mendeley_asl", "paths.data.raw.mendeley_asl")
        ),
        raw_webcam_real_test=_resolve_under_root(
            root,
            _require_str(
                raw_data, "webcam_real_test", "paths.data.raw.webcam_real_test"
            ),
        ),
        processed_dir=_resolve_under_root(
            root, _require_str(processed, "dir", "paths.data.processed.dir")
        ),
        features_parquet=_resolve_under_root(
            root, _require_str(processed, "features", "paths.data.processed.features")
        ),
        train_parquet=_resolve_under_root(
            root, _require_str(processed, "train", "paths.data.processed.train")
        ),
        val_parquet=_resolve_under_root(
            root, _require_str(processed, "val", "paths.data.processed.val")
        ),
        test_parquet=_resolve_under_root(
            root, _require_str(processed, "test", "paths.data.processed.test")
        ),
        metadata_dir=_resolve_under_root(
            root, _require_str(data, "metadata", "paths.data.metadata")
        ),
        checkpoints_dir=_resolve_under_root(
            root, _require_str(models, "checkpoints", "paths.models.checkpoints")
        ),
        production_dir=_resolve_under_root(
            root, _require_str(models, "production", "paths.models.production")
        ),
        figures_dir=_resolve_under_root(
            root, _require_str(reports, "figures", "paths.reports.figures")
        ),
        metrics_dir=_resolve_under_root(
            root, _require_str(reports, "metrics", "paths.reports.metrics")
        ),
        logs_dir=_resolve_under_root(
            root, _require_str(reports, "logs", "paths.reports.logs")
        ),
        label_map=_resolve_under_root(
            root, _require_str(configs, "label_map", "paths.configs.label_map")
        ),
        feature_config=_resolve_under_root(
            root, _require_str(configs, "feature_config", "paths.configs.feature_config")
        ),
    )


def _parse_mediapipe(raw: Mapping[str, Any]) -> MediaPipeConfig:
    block = _require_mapping(raw, "mediapipe", "mediapipe")
    complexity = _require_int(block, "model_complexity", "mediapipe.model_complexity")
    if complexity not in (0, 1):
        raise ConfigError(
            "mediapipe.model_complexity chỉ nhận 0 hoặc 1 "
            f"(giá trị hiện tại: {complexity})."
        )
    detection = _require_unit_interval(
        block, "min_detection_confidence", "mediapipe.min_detection_confidence"
    )
    tracking = _require_unit_interval(
        block, "min_tracking_confidence", "mediapipe.min_tracking_confidence"
    )
    max_hands = _require_int(block, "max_num_hands", "mediapipe.max_num_hands")
    if max_hands < 1:
        raise ConfigError("mediapipe.max_num_hands phải >= 1.")
    return MediaPipeConfig(
        model_complexity=complexity,
        min_detection_confidence=detection,
        min_tracking_confidence=tracking,
        max_num_hands=max_hands,
    )


def _parse_split(raw: Mapping[str, Any]) -> SplitConfig:
    block = _require_mapping(raw, "split", "split")
    train_ratio = _require_float(block, "train_ratio", "split.train_ratio")
    val_ratio = _require_float(block, "val_ratio", "split.val_ratio")
    test_ratio = _require_float(block, "test_ratio", "split.test_ratio")
    for name, ratio in (
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ):
        if ratio <= 0.0 or ratio >= 1.0:
            raise ConfigError(f"split.{name} phải nằm trong (0, 1); nhận {ratio}.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        raise ConfigError(
            "Tổng train_ratio + val_ratio + test_ratio phải bằng 1; "
            f"hiện tại là {total:.6f}."
        )
    seed = _require_int(block, "random_seed", "split.random_seed")
    if seed < 0:
        raise ConfigError("split.random_seed phải >= 0.")
    return SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
    )


def _parse_mlp(raw: Mapping[str, Any]) -> MLPConfig:
    training = _require_mapping(raw, "training", "training")
    block = _require_mapping(training, "mlp", "training.mlp")
    input_dim = _require_int(block, "input_dim", "training.mlp.input_dim")
    num_classes = _require_int(block, "num_classes", "training.mlp.num_classes")
    batch_size = _require_int(block, "batch_size", "training.mlp.batch_size")
    epochs = _require_int(block, "epochs", "training.mlp.epochs")
    learning_rate = _require_float(block, "learning_rate", "training.mlp.learning_rate")
    early = _require_int(
        block, "early_stopping_patience", "training.mlp.early_stopping_patience"
    )
    lr_patience = _require_int(block, "lr_patience", "training.mlp.lr_patience")
    if input_dim <= 0 or num_classes <= 0 or batch_size <= 0 or epochs <= 0:
        raise ConfigError("input_dim, num_classes, batch_size và epochs phải là số nguyên dương.")
    if learning_rate <= 0.0:
        raise ConfigError("training.mlp.learning_rate phải > 0.")
    if early < 1 or lr_patience < 1:
        raise ConfigError("early_stopping_patience và lr_patience phải >= 1.")
    return MLPConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early,
        lr_patience=lr_patience,
    )


def _parse_postprocess(raw: Mapping[str, Any]) -> PostprocessConfig:
    inference = _require_mapping(raw, "inference", "inference")
    block = _require_mapping(inference, "postprocess", "inference.postprocess")
    ema_alpha = _require_unit_interval(block, "ema_alpha", "postprocess.ema_alpha")
    if ema_alpha <= 0.0:
        raise ConfigError("postprocess.ema_alpha phải nằm trong (0, 1].")
    conf_threshold = _require_unit_interval(
        block, "conf_threshold", "postprocess.conf_threshold"
    )
    margin_threshold = _require_unit_interval(
        block, "margin_threshold", "postprocess.margin_threshold"
    )
    window_size = _require_int(block, "window_size", "postprocess.window_size")
    min_votes = _require_int(block, "min_votes", "postprocess.min_votes")
    hold_seconds = _require_float(block, "hold_seconds", "postprocess.hold_seconds")
    if window_size < 1:
        raise ConfigError("postprocess.window_size phải >= 1.")
    if min_votes < 1 or min_votes > window_size:
        raise ConfigError(
            "postprocess.min_votes phải thỏa 1 <= min_votes <= window_size "
            f"(nhận min_votes={min_votes}, window_size={window_size})."
        )
    if hold_seconds <= 0.0:
        raise ConfigError("postprocess.hold_seconds phải > 0.")
    return PostprocessConfig(
        ema_alpha=ema_alpha,
        conf_threshold=conf_threshold,
        margin_threshold=margin_threshold,
        window_size=window_size,
        min_votes=min_votes,
        hold_seconds=hold_seconds,
    )


def _parse_index_to_label(raw_map: Mapping[Any, Any], path: Path) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for key, value in raw_map.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ConfigError(f"Khóa nhãn không hợp lệ trong {path}: {key!r}.")
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Khóa nhãn không phải số trong {path}: {key!r}.") from exc
        if not isinstance(value, str) or len(value) != 1:
            raise ConfigError(f"Nhãn tại {index} phải là một chữ cái: {value!r}.")
        parsed[index] = value.upper()
    expected = set(range(26))
    if set(parsed) != expected:
        missing = sorted(expected - set(parsed))
        extra = sorted(set(parsed) - expected)
        raise ConfigError(
            f"{path} phải ánh xạ đúng 0..25. Thiếu {missing}, thừa {extra}."
        )
    labels = [parsed[index] for index in range(26)]
    expected_labels = [chr(ord("A") + index) for index in range(26)]
    if labels != expected_labels:
        raise ConfigError(
            f"{path} phải ánh xạ 0->A ... 25->Z, nhận {labels[0]}..{labels[-1]}."
        )
    return parsed


def _parse_landmarks(payload: Mapping[str, Any], path: Path) -> tuple[LandmarkSpec, ...]:
    raw_items = _require_list(payload, "landmarks", "feature.landmarks")
    landmarks: list[LandmarkSpec] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ConfigError(f"Landmark trong {path} phải là object.")
        landmarks.append(
            LandmarkSpec(
                index=_require_int(item, "index", "feature.landmarks.index"),
                name=_require_str(item, "name", "feature.landmarks.name"),
            )
        )
    indexes = [item.index for item in landmarks]
    if indexes != list(range(_LANDMARK_COUNT)):
        raise ConfigError(
            f"{path} phải khai báo đúng landmark 0..20 theo thứ tự MediaPipe."
        )
    return tuple(landmarks)


def _parse_bones(
    group_payload: Mapping[str, Any], group: FeatureGroupSpec, path: Path
) -> tuple[BoneSpec, ...]:
    raw_items = _require_list(group_payload, "connections", f"{group.name}.connections")
    bones: list[BoneSpec] = []
    for offset, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ConfigError(f"Connection trong {path} phải là object.")
        bone = BoneSpec(
            feature_index=_require_int(item, "feature_index", "bone.feature_index"),
            start=_require_int(item, "start", "bone.start"),
            end=_require_int(item, "end", "bone.end"),
            name=_require_str(item, "name", "bone.name"),
        )
        _validate_landmark_index(bone.start, path)
        _validate_landmark_index(bone.end, path)
        expected_index = group.start + offset
        if bone.feature_index != expected_index:
            raise ConfigError(
                f"Xương {bone.name} phải có feature_index={expected_index}, "
                f"nhận {bone.feature_index}."
            )
        bones.append(bone)
    if len(bones) != group.size:
        raise ConfigError(
            f"Nhóm {group.name} khai báo size={group.size} nhưng có {len(bones)} xương."
        )
    return tuple(bones)


def _parse_angles(
    group_payload: Mapping[str, Any], group: FeatureGroupSpec, path: Path
) -> tuple[AngleSpec, ...]:
    raw_items = _require_list(group_payload, "angles", f"{group.name}.angles")
    angles: list[AngleSpec] = []
    for offset, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ConfigError(f"Angle trong {path} phải là object.")
        angle = AngleSpec(
            feature_index=_require_int(item, "feature_index", "angle.feature_index"),
            name=_require_str(item, "name", "angle.name"),
            vertex=_require_int(item, "vertex", "angle.vertex"),
            arm_a=_require_int(item, "arm_a", "angle.arm_a"),
            arm_b=_require_int(item, "arm_b", "angle.arm_b"),
            description=_require_str(item, "description", "angle.description"),
        )
        for landmark_id in (angle.vertex, angle.arm_a, angle.arm_b):
            _validate_landmark_index(landmark_id, path)
        if len({angle.vertex, angle.arm_a, angle.arm_b}) < 3:
            raise ConfigError(f"Góc {angle.name} phải dùng ba landmark phân biệt.")
        expected_index = group.start + offset
        if angle.feature_index != expected_index:
            raise ConfigError(
                f"Góc {angle.name} phải có feature_index={expected_index}, "
                f"nhận {angle.feature_index}."
            )
        angles.append(angle)
    if len(angles) != group.size:
        raise ConfigError(
            f"Nhóm {group.name} khai báo size={group.size} nhưng có {len(angles)} góc."
        )
    return tuple(angles)


def _validate_landmark_index(index: int, path: Path) -> None:
    if index < 0 or index >= _LANDMARK_COUNT:
        raise ConfigError(f"Chỉ số landmark {index} nằm ngoài 0..20 trong {path}.")


def _validate_feature_schema(schema: FeatureSchema, path: Path) -> None:
    if schema.input_dim <= 0:
        raise ConfigError("feature.input_dim phải > 0.")
    if schema.origin_landmark_index != 0:
        raise ConfigError("Gốc tọa độ phải là landmark cổ tay (0).")
    if not 0 <= schema.scale_landmark_index < _LANDMARK_COUNT:
        raise ConfigError("scale_landmark_index nằm ngoài 0..20.")
    if schema.scale_epsilon <= 0.0:
        raise ConfigError("scale_epsilon phải > 0.")
    cursor = 0
    for group in schema.groups:
        if group.start != cursor:
            raise ConfigError(
                f"Nhóm {group.name} phải bắt đầu tại {cursor}, nhận {group.start} ({path})."
            )
        cursor = group.end
    if cursor != schema.input_dim:
        raise ConfigError(
            f"Các nhóm phủ [0, {cursor}) nhưng input_dim={schema.input_dim} ({path})."
        )
    names = [group.name for group in schema.groups]
    if names not in (
        ["normalized_landmarks", "bone_lengths", "joint_angles"],
        ["normalized_landmarks", "finger_angles", "tip_distances", "palm_normal"],
    ):
        raise ConfigError(
            "feature.groups không khớp hợp đồng đặc trưng hình học. "
            f"Nhận {names}."
        )
    if names == ["normalized_landmarks", "bone_lengths", "joint_angles"]:
        if not schema.bones or not schema.angles:
            raise ConfigError(f"Thiếu danh sách xương hoặc góc trong {path}.")


def _validate_cross_refs(config: ProjectConfig) -> None:
    if config.mlp.input_dim != config.features.input_dim:
        raise ConfigError(
            "training.mlp.input_dim "
            f"({config.mlp.input_dim}) khác feature.input_dim "
            f"({config.features.input_dim})."
        )
    if config.mlp.num_classes != len(config.index_to_label):
        raise ConfigError(
            "training.mlp.num_classes "
            f"({config.mlp.num_classes}) khác số nhãn ({len(config.index_to_label)})."
        )
    if config.features.input_dim != 91:
        raise ConfigError(
            "Hợp đồng đồ án yêu cầu đúng 91 đặc trưng, "
            f"nhận {config.features.input_dim}."
        )


def _require_mapping(data: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    value = _require_key(data, key, path)
    if not isinstance(value, dict):
        raise ConfigError(f"{path} phải là mapping.")
    return value


def _require_list(data: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = _require_key(data, key, path)
    if not isinstance(value, list):
        raise ConfigError(f"{path} phải là danh sách.")
    return value


def _require_str(data: Mapping[str, Any], key: str, path: str) -> str:
    value = _require_key(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} phải là chuỗi khác rỗng.")
    return value


def _require_int(data: Mapping[str, Any], key: str, path: str) -> int:
    value = _require_key(data, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} phải là số nguyên, nhận {value!r}.")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigError(f"{path} phải là số nguyên, nhận {value!r}.")
    return int(value)


def _require_float(data: Mapping[str, Any], key: str, path: str) -> float:
    value = _require_key(data, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} phải là số, nhận {value!r}.")
    return float(value)


def _require_unit_interval(data: Mapping[str, Any], key: str, path: str) -> float:
    value = _require_float(data, key, path)
    if value < 0.0 or value > 1.0:
        raise ConfigError(f"{path} phải nằm trong [0, 1]; nhận {value}.")
    return value


def _require_key(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise ConfigError(f"Thiếu khóa bắt buộc '{path}'.")
    return data[key]
