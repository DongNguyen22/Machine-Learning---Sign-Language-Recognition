"""Train a LinearSVC baseline on HOG features using a validation split.

The provided test split is intentionally not loaded here. It must remain
untouched until the model settings (including C) have been selected using
validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


EXPECTED_FEATURE_COUNT = 1764
EXPECTED_CLASS_COUNT = 36
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

IMAGE_SIZE = (64, 64)
HOG_PARAMETERS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LinearSVC C=1.0 baseline with a validation split"
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/svm_hog_features.npz"),
        help="Path to the compressed HOG feature file",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/svm/svm_baseline_c1.joblib"),
        help="Output path for the baseline model bundle",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("results/svm/svm_baseline_c1_validation.json"),
        help="Output path for validation metrics",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="LinearSVC regularization parameter (baseline default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="Maximum number of LinearSVC iterations",
    )
    parser.add_argument(
        "--split-indices",
        type=Path,
        default=None,
        help=(
            "Optional path to a saved split .npz file containing fit_indices "
            "and validation_indices. When omitted, uses the original random "
            "stratified split."
        ),
    )
    return parser.parse_args()


def sha256_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_saved_split(
    split_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Split indices file not found: {split_path}")

    with np.load(split_path, allow_pickle=False) as split_data:
        required = {"fit_indices", "validation_indices", "X_train_sha256", "y_train_sha256"}
        missing = required.difference(split_data.files)
        if missing:
            raise ValueError(f"Split file is missing keys: {sorted(missing)}")

        expected_x_sha = str(split_data["X_train_sha256"].item())
        expected_y_sha = str(split_data["y_train_sha256"].item())
        actual_x_sha = sha256_array(X_train)
        actual_y_sha = sha256_array(y_train)
        if expected_x_sha != actual_x_sha or expected_y_sha != actual_y_sha:
            raise ValueError(
                "X_train or y_train SHA-256 fingerprint does not match the saved split file."
            )

        fit_indices = split_data["fit_indices"].astype(np.int32).copy()
        val_indices = split_data["validation_indices"].astype(np.int32).copy()
        split_type = (
            str(split_data["split_type"].item())
            if "split_type" in split_data.files
            else "custom_saved_split"
        )
        fit_groups = (
            split_data["fit_groups"].astype(str).tolist()
            if "fit_groups" in split_data.files
            else None
        )
        val_groups = (
            split_data["validation_groups"].astype(str).tolist()
            if "validation_groups" in split_data.files
            else None
        )

    fit_set = set(fit_indices.tolist())
    val_set = set(val_indices.tolist())
    if fit_set.intersection(val_set):
        raise ValueError("Saved split has overlapping fit and validation indices.")
    if fit_set.union(val_set) != set(range(len(X_train))):
        raise ValueError("Saved split indices do not cover all rows of X_train.")

    info: dict[str, object] = {
        "split_type": split_type,
        "split_file": str(split_path),
        "fit_groups": fit_groups,
        "validation_groups": val_groups,
    }
    return fit_indices, val_indices, info


def load_training_data(features_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    with np.load(features_path, allow_pickle=False) as data:
        required_keys = {"X_train", "y_train", "class_names"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Feature file is missing keys: {missing}")

        # Copy the arrays before the npz file is closed.
        X_train = data["X_train"].copy()
        y_train = data["y_train"].copy()
        class_names = data["class_names"].astype(str).copy()

    validate_training_data(X_train, y_train, class_names)
    return X_train, y_train, class_names


def validate_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: np.ndarray,
) -> None:
    if X_train.ndim != 2 or X_train.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "X_train must have shape (number_of_images, 1764), "
            f"but got {X_train.shape}"
        )
    if y_train.ndim != 1 or len(y_train) != len(X_train):
        raise ValueError(
            "y_train must be one-dimensional and match X_train; "
            f"got X_train={X_train.shape}, y_train={y_train.shape}"
        )
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected 36 class names, but got {len(class_names)}"
        )
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values")

    expected_labels = np.arange(len(class_names))
    actual_labels = np.unique(y_train)
    if not np.array_equal(actual_labels, expected_labels):
        raise ValueError(
            "Training labels must contain every integer from 0 to 35; "
            f"got {actual_labels.tolist()}"
        )


def build_pipeline(c_value: float, max_iter: int) -> Pipeline:
    if c_value <= 0:
        raise ValueError(f"C must be greater than 0, but got {c_value}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be greater than 0, but got {max_iter}")

    # HOG already applies L2-Hys block normalization. This first baseline does
    # not add a scaler, but a Pipeline keeps preprocessing/model steps grouped
    # and prevents leakage if a scaler is added in a later experiment.
    return Pipeline(
        steps=[
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    random_state=RANDOM_STATE,
                    max_iter=max_iter,
                    dual="auto",
                ),
            )
        ]
    )


def main() -> None:
    args = parse_args()

    print(f"Loading HOG features from: {args.features.resolve()}")
    X_train, y_train, class_names = load_training_data(args.features)
    print(f"Full training set: X={X_train.shape}, y={y_train.shape}")
    print("The saved test set is not loaded or used by this baseline.")

    if args.split_indices is not None:
        fit_indices, val_indices, split_info = load_saved_split(
            args.split_indices, X_train, y_train
        )
        X_fit = X_train[fit_indices]
        y_fit = y_train[fit_indices]
        X_validation = X_train[val_indices]
        y_validation = y_train[val_indices]
        print(f"Loaded saved split ({split_info['split_type']}) from: {args.split_indices}")
        if split_info["fit_groups"] is not None:
            print(f"Fit prefix groups       : {split_info['fit_groups']}")
            print(f"Validation prefix groups: {split_info['validation_groups']}")
    else:
        X_fit, X_validation, y_fit, y_validation = train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
        split_info = {
            "split_type": "random_stratified_image_split",
            "split_file": None,
            "fit_groups": None,
            "validation_groups": None,
        }
    print(f"Fit split: X={X_fit.shape}, y={y_fit.shape}")
    print(
        "Validation split: "
        f"X={X_validation.shape}, y={y_validation.shape}"
    )

    pipeline = build_pipeline(args.c, args.max_iter)

    print(f"Training LinearSVC baseline with C={args.c}...")
    train_start = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    training_seconds = time.perf_counter() - train_start

    y_fit_pred = pipeline.predict(X_fit)
    fit_accuracy = accuracy_score(y_fit, y_fit_pred)
    fit_correct = int(np.sum(y_fit_pred == y_fit))
    fit_incorrect = int(np.sum(y_fit_pred != y_fit))

    prediction_start = time.perf_counter()
    y_pred = pipeline.predict(X_validation)
    prediction_seconds = time.perf_counter() - prediction_start

    val_correct = int(np.sum(y_pred == y_validation))
    val_incorrect = int(np.sum(y_pred != y_validation))
    accuracy = accuracy_score(y_validation, y_pred)
    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )
    labels = np.arange(len(class_names))
    report_by_class = classification_report(
        y_validation,
        y_pred,
        labels=labels,
        target_names=class_names.tolist(),
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "experiment": (
            "linear_svc_baseline_prefix_group_validation"
            if split_info["split_type"] != "random_stratified_image_split"
            else "linear_svc_baseline_validation"
        ),
        "split_info": split_info,
        "c": args.c,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "full_train_samples": int(len(X_train)),
        "fit_samples": int(len(X_fit)),
        "fit_correct_predictions": fit_correct,
        "fit_incorrect_predictions": fit_incorrect,
        "fit_accuracy": float(fit_accuracy),
        "validation_samples": int(len(X_validation)),
        "validation_correct_predictions": val_correct,
        "validation_incorrect_predictions": val_incorrect,
        "feature_count": int(X_train.shape[1]),
        "class_count": int(len(class_names)),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "training_seconds": training_seconds,
        "validation_prediction_seconds": prediction_seconds,
        "classification_report": report_by_class,
        "test_set_used": False,
        "limitation": (
            "Prefix-group validation separates filename prefixes (P1..P10) between "
            "fit and validation, but because P1..P10 are not verified via local metadata "
            "as unique signers (and the shared test set still contains P1..P10), "
            "results represent prefix-group generalization rather than verified new-signer generalization."
        ),
    }

    model_bundle = {
        "model": pipeline,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "hog_parameters": HOG_PARAMETERS,
        "c": args.c,
        "random_state": RANDOM_STATE,
        "validation_size": VALIDATION_SIZE,
        "split_info": split_info,
        "fit_metrics": {
            "accuracy": float(fit_accuracy),
            "correct_predictions": fit_correct,
            "incorrect_predictions": fit_incorrect,
        },
        "validation_metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "correct_predictions": val_correct,
            "incorrect_predictions": val_incorrect,
        },
        "training_scope": (
            f"fit split ({len(X_fit)} of {len(X_train)} train samples; "
            f"split_type={split_info['split_type']}); test set not used"
        ),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, args.model_output)
    args.report_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== BASELINE VALIDATION RESULT ==========")
    print(f"Accuracy        : {accuracy:.6f}")
    print(f"Macro precision : {macro_precision:.6f}")
    print(f"Macro recall    : {macro_recall:.6f}")
    print(f"Macro F1        : {macro_f1:.6f}")
    print(f"Training time   : {training_seconds:.3f} seconds")
    print(f"Prediction time : {prediction_seconds:.3f} seconds")
    print(f"Model saved to  : {args.model_output.resolve()}")
    print(f"Report saved to : {args.report_output.resolve()}")


if __name__ == "__main__":
    main()
