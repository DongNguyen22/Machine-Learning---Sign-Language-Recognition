"""Evaluate all saved ASL models on in-domain and webcam test sets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_fps import benchmark_model, model_size
from src.evaluation.error_analyzer import (
    extract_top_confusions,
    plot_learning_curves,
    plot_normalized_confusion_matrix,
)
from src.evaluation.metrics_calculator import calculate_metrics, per_class_dataframe
from src.utils.config_loader import load_config

ASL_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def feature_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu tập dữ liệu: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise ValueError(f"Tập dữ liệu trống: {path}")
    if "label" not in frame.columns:
        raise ValueError(f"Thiếu cột label trong: {path}")
    return frame


def extract_xy(frame: pd.DataFrame, input_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = sorted(
        (column for column in frame.columns if column.startswith("feature_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if input_dim is not None:
        feature_columns = feature_columns[:input_dim]
        if len(feature_columns) != input_dim:
            raise ValueError(
                f"Test có {len(feature_columns)} đặc trưng, metadata yêu cầu {input_dim}."
            )
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame["label"].astype(str).str.upper().to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Tập test chứa NaN hoặc Inf.")
    if set(np.unique(y)) - set(ASL_LABELS):
        raise ValueError("Tập test chứa nhãn ngoài A-Z.")
    return x, y


def load_artifact(path: Path) -> object:
    suffix = path.suffix.lower()
    if suffix == ".keras":
        from tensorflow import keras
        return keras.models.load_model(path)
    if suffix == ".joblib":
        return joblib.load(path)
    if suffix == ".onnx":
        import onnxruntime as ort
        return ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
    raise ValueError(f"Định dạng mô hình không hỗ trợ: {suffix}")


def predict_artifact(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "get_inputs") and hasattr(model, "run"):
        input_name = model.get_inputs()[0].name
        output = model.run(None, {input_name: x.astype(np.float32)})
        probabilities = np.asarray(output[0])
        if probabilities.ndim == 1:
            return probabilities.astype(np.int32)
        return np.argmax(probabilities, axis=1)

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x))
        if probabilities.shape[1] == 26:
            return np.argmax(probabilities, axis=1)
        return np.asarray(model.predict(x))

    if hasattr(model, "predict"):
        output = model.predict(x, verbose=0)
        output = np.asarray(output)
        if output.ndim == 2:
            return np.argmax(output, axis=1)
        return output.astype(np.int32)

    raise TypeError("Mô hình không có hàm predict được hỗ trợ.")


def decode_predictions(
    predictions: np.ndarray,
    label_to_index: dict[str, int],
    model_path: Path,
) -> np.ndarray:
    if model_path.suffix.lower() == ".joblib":
        index_to_label = {int(index): str(label) for label, index in label_to_index.items()}
        return np.asarray([index_to_label.get(int(value), "A") for value in predictions])
    return np.asarray([ASL_LABELS[int(value)] for value in predictions])


def artifact_paths(production_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in production_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".keras", ".joblib", ".onnx"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved ASL models")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Số vòng suy luận CPU cho benchmark",
    )
    args = parser.parse_args()

    config = load_config()
    production_dir = ROOT / "models" / "production"
    processed_dir = ROOT / "datasets" / "processed"
    figures_dir = ROOT / "reports" / "figures"
    metrics_dir = ROOT / "reports" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    production_dir.mkdir(parents=True, exist_ok=True)
    metadata_paths = sorted(production_dir.glob("*_metadata.json"))
    models = artifact_paths(production_dir)
    if not metadata_paths:
        print(
            "WARNING: No *_metadata.json found in models/production; "
            "falling back to model-file discovery."
        )
    test_sets = (
        ("test_in_domain", processed_dir / "test_in_domain_features.parquet"),
        ("test_webcam", processed_dir / "test_webcam_features.parquet"),
    )

    summary_rows: list[dict[str, object]] = []
    if not models:
        print("No models found in models/production.")

    evaluation_artifacts: list[tuple[dict[str, object], Path]] = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_path = Path(str(metadata.get("model_path", "")))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if not model_path.is_file():
            print(
                f"WARNING: Missing model referenced by {metadata_path.name}: "
                f"{model_path}"
            )
            continue
        evaluation_artifacts.append((metadata, model_path))
        onnx_path_value = metadata.get("onnx_path")
        if onnx_path_value:
            onnx_path = Path(str(onnx_path_value))
            if not onnx_path.is_absolute():
                onnx_path = ROOT / onnx_path
            if onnx_path.is_file():
                evaluation_artifacts.append((metadata, onnx_path))
            else:
                print(
                    f"WARNING: Missing ONNX model referenced by "
                    f"{metadata_path.name}: {onnx_path}"
                )
    if not metadata_paths:
        evaluation_artifacts = [
            ({}, model_path) for model_path in models
        ]

    for metadata, model_path in evaluation_artifacts:
        experiment_name = str(metadata.get("experiment_name", model_path.stem))
        input_dim = int(metadata.get("input_dim", 91))
        scaler_path = metadata.get("scaler_path")
        if scaler_path:
            print(
                f"{experiment_name}: scaler/model pipeline: {scaler_path}"
            )
        model = load_artifact(model_path)
        for test_name, test_path in test_sets:
            try:
                test_frame = feature_frame(test_path)
                if len(test_frame) == 0:
                    print(
                        f"WARNING: Test dataset is empty (0 rows): "
                        f"{test_path}"
                    )
                    continue
                x, y_true = extract_xy(test_frame, input_dim=input_dim)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping {experiment_name}/{test_name}: {exc}")
                continue

            raw_predictions = predict_artifact(model, x)
            y_pred = decode_predictions(raw_predictions, config.label_to_index, model_path)

            metrics = calculate_metrics(y_true, y_pred)
            cm_path = figures_dir / f"{experiment_name}_{test_name}_confusion_matrix.png"
            plot_normalized_confusion_matrix(
                y_true,
                y_pred,
                cm_path,
                title=f"{experiment_name} - {test_name}",
            )
            top_confusions = extract_top_confusions(y_true, y_pred)
            top_path = metrics_dir / f"{experiment_name}_{test_name}_top_confusions.csv"
            pd.DataFrame(top_confusions).to_csv(top_path, index=False)

            per_class = per_class_dataframe(metrics)
            per_class_path = metrics_dir / f"{experiment_name}_{test_name}_per_class.csv"
            per_class.to_csv(per_class_path, index=False)

            size = model_size(model_path)
            benchmark = benchmark_model(
                model,
                x[0],
                warmup=50,
                iterations=args.iterations,
            )
            summary_rows.append({
                "experiment": experiment_name,
                "test_set": test_name,
                "model_format": model_path.suffix.lower(),
                "model_size_mb": size["mb"],
                "model_size_kb": size["kb"],
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "inference_ms": benchmark["latency_ms"],
                "fps": benchmark["fps"],
            })

        for history_path in sorted(metrics_dir.glob(f"{experiment_name}_history.json")):
            plot_learning_curves(history_path, figures_dir, experiment_name)

    summary_path = metrics_dir / "benchmark_summary.csv"
    summary = pd.DataFrame(summary_rows)
    summary_columns = [
        "experiment", "test_set", "model_format", "model_size_mb",
        "model_size_kb", "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "inference_ms", "fps",
    ]
    if summary.empty:
        summary = pd.DataFrame(columns=summary_columns)
    else:
        summary = summary[summary_columns]
    summary.to_csv(summary_path, index=False)

    print(json.dumps({
        "models": [str(path) for path in models],
        "summary": str(summary_path),
        "rows": len(summary),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
