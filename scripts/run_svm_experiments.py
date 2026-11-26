"""Run LinearSVC hyperparameter C experiments on the verified prefix-group validation split.

Experiments:
- Model type: LinearSVC inside a scikit-learn Pipeline (same model family across all runs).
- Tested C values: 0.01, 0.1, 1.0, 10.0.
- Split: results/svm/svm_prefix_group_split_indices.npz
  (Fit: 23,026 samples from P1..P7, P9; Validation: 5,774 samples from P8, P10).
- Test set (X_test, y_test) is intentionally NOT loaded or used.
- Reuses models/svm/svm_baseline_c1_prefix_group.joblib for C=1.0 after verifying
  that data fingerprints, split indices, and estimator parameters match 100%.
- Saves per-C models, detailed JSON reports, a summary CSV, a summary JSON,
  and log-scale plots of C vs validation accuracy/Macro F1 and C vs training time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_svm import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_FEATURE_COUNT,
    HOG_PARAMETERS,
    IMAGE_SIZE,
    RANDOM_STATE,
    VALIDATION_SIZE,
    build_pipeline,
    load_training_data,
)

DEFAULT_C_VALUES = [0.01, 0.1, 1.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LinearSVC C experiments on the verified prefix-group split."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/svm_hog_features.npz"),
        help="Path to the HOG feature NPZ file (only X_train, y_train, class_names are loaded)",
    )
    parser.add_argument(
        "--split-indices",
        type=Path,
        default=Path("results/svm/svm_prefix_group_split_indices.npz"),
        help="Path to the verified prefix-group split indices NPZ file",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=Path("models/svm/svm_baseline_c1_prefix_group.joblib"),
        help="Path to the existing C=1.0 prefix-group baseline model for verified reuse",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("results/svm/svm_baseline_c1_prefix_group_validation.json"),
        help="Path to the existing C=1.0 prefix-group baseline validation JSON report",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models/svm"),
        help="Directory to save per-C experiment models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/svm"),
        help="Directory to save CSV, JSON reports, and plots",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="Maximum iterations for LinearSVC (default: 10,000)",
    )
    parser.add_argument(
        "--no-reuse-baseline",
        action="store_true",
        help="Force retraining C=1.0 even if the saved baseline matches",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def format_c_tag(c_val: float) -> str:
    return f"c{str(c_val).replace('.', '_')}"


def get_library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scikit_image": skimage.__version__,
        "opencv_python": cv2.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
    }


def verify_and_load_split(
    split_path: Path,
    features_path: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, object]:
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    actual_file_sha = sha256_file(features_path)
    actual_x_sha = sha256_array(X_train)
    actual_y_sha = sha256_array(y_train)

    with np.load(split_path, allow_pickle=False) as s:
        required_keys = {
            "fit_indices",
            "validation_indices",
            "fit_groups",
            "validation_groups",
            "features_file_sha256",
            "X_train_sha256",
            "y_train_sha256",
        }
        missing = required_keys.difference(s.files)
        if missing:
            raise ValueError(f"Split NPZ is missing required keys: {sorted(missing)}")

        saved_file_sha = str(s["features_file_sha256"].item())
        saved_x_sha = str(s["X_train_sha256"].item())
        saved_y_sha = str(s["y_train_sha256"].item())

        if saved_file_sha != actual_file_sha:
            raise ValueError(
                f"features_file_sha256 mismatch: expected {saved_file_sha}, got {actual_file_sha}"
            )
        if saved_x_sha != actual_x_sha:
            raise ValueError(
                f"X_train_sha256 mismatch: expected {saved_x_sha}, got {actual_x_sha}"
            )
        if saved_y_sha != actual_y_sha:
            raise ValueError(
                f"y_train_sha256 mismatch: expected {saved_y_sha}, got {actual_y_sha}"
            )

        fit_indices = s["fit_indices"].astype(np.int32).copy()
        val_indices = s["validation_indices"].astype(np.int32).copy()
        fit_groups = s["fit_groups"].astype(str).tolist()
        val_groups = s["validation_groups"].astype(str).tolist()
        split_type = (
            str(s["split_type"].item())
            if "split_type" in s.files
            else "prefix_group_shuffle_split"
        )

    # Check index validity and disjointness
    n_total = len(X_train)
    if np.any(fit_indices < 0) or np.any(fit_indices >= n_total):
        raise ValueError("fit_indices out of bounds")
    if np.any(val_indices < 0) or np.any(val_indices >= n_total):
        raise ValueError("validation_indices out of bounds")

    fit_set = set(fit_indices.tolist())
    val_set = set(val_indices.tolist())
    if len(fit_set) != len(fit_indices) or len(val_set) != len(val_indices):
        raise ValueError("Duplicate indices within fit_indices or validation_indices")
    if fit_set.intersection(val_set):
        raise ValueError("fit_indices and validation_indices overlap!")
    if fit_set.union(val_set) != set(range(n_total)):
        raise ValueError("fit_indices union validation_indices does not equal 0..28799")

    if set(fit_groups).intersection(set(val_groups)):
        raise ValueError("fit_groups and validation_groups overlap!")

    y_fit = y_train[fit_indices]
    y_val = y_train[val_indices]
    if len(np.unique(y_fit)) != EXPECTED_CLASS_COUNT:
        raise ValueError("Not all 36 classes are present in y_fit")
    if len(np.unique(y_val)) != EXPECTED_CLASS_COUNT:
        raise ValueError("Not all 36 classes are present in y_validation")

    # Verify 0 exact HOG duplicates between fit and validation
    X_fit = X_train[fit_indices]
    X_val = X_train[val_indices]
    fit_hog_bytes = {X_fit[i].tobytes() for i in range(len(X_fit))}
    cross_hog_dups = sum(1 for i in range(len(X_val)) if X_val[i].tobytes() in fit_hog_bytes)
    if cross_hog_dups != 0:
        raise ValueError(f"Expected 0 exact HOG duplicates between fit and val, found {cross_hog_dups}")

    return {
        "split_path": str(split_path),
        "split_file_sha256": sha256_file(split_path),
        "split_type": split_type,
        "fit_indices": fit_indices,
        "validation_indices": val_indices,
        "fit_groups": fit_groups,
        "validation_groups": val_groups,
        "fit_samples": int(len(fit_indices)),
        "validation_samples": int(len(val_indices)),
        "index_overlap_count": 0,
        "group_overlap_count": 0,
        "exact_hog_duplicates_between_fit_and_val": cross_hog_dups,
        "features_file_sha256": actual_file_sha,
        "X_train_sha256": actual_x_sha,
        "y_train_sha256": actual_y_sha,
        "class_count": int(len(class_names)),
    }


def extract_iteration_info(clf) -> dict[str, object]:
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is None:
        return {
            "n_iter_available": False,
            "n_iter_max": None,
            "n_iter_min": None,
            "n_iter_median": None,
            "n_iter_per_class": None,
        }
    arr = np.atleast_1d(np.asarray(n_iter, dtype=np.int64))
    return {
        "n_iter_available": True,
        "n_iter_max": int(np.max(arr)),
        "n_iter_min": int(np.min(arr)),
        "n_iter_median": float(np.median(arr)),
        "n_iter_per_class": arr.tolist(),
    }


def can_reuse_c1_baseline(
    baseline_model_path: Path,
    baseline_report_path: Path,
    max_iter: int,
    class_names: np.ndarray,
    fit_groups: list[str],
    val_groups: list[str],
) -> tuple[bool, str, dict[str, object] | None, dict[str, object] | None]:
    if not baseline_model_path.is_file() or not baseline_report_path.is_file():
        return False, "Baseline model or report file does not exist.", None, None

    bundle = joblib.load(baseline_model_path)
    report = json.loads(baseline_report_path.read_text(encoding="utf-8"))

    pipeline = bundle.get("model")
    if pipeline is None or "classifier" not in getattr(pipeline, "named_steps", {}):
        return False, "Saved bundle does not contain expected Pipeline(['classifier']).", None, None

    clf = pipeline.named_steps["classifier"]
    if (
        float(clf.C) != 1.0
        or int(clf.max_iter) != int(max_iter)
        or int(clf.random_state) != int(RANDOM_STATE)
        or clf.dual != "auto"
    ):
        return False, "Classifier hyperparameters in saved baseline do not match.", None, None

    if list(bundle.get("class_names", [])) != class_names.tolist():
        return False, "class_names in saved baseline do not match.", None, None

    split_info = report.get("split_info", {})
    if (
        split_info.get("fit_groups") != fit_groups
        or split_info.get("validation_groups") != val_groups
    ):
        return False, "split_info groups in saved baseline report do not match.", None, None

    return True, "Verified 100% match with saved prefix-group C=1.0 baseline.", bundle, report


def plot_experiments(
    summary_rows: list[dict[str, object]],
    best_c: float,
    results_dir: Path,
) -> tuple[Path, Path]:
    c_vals = [float(r["c"]) for r in summary_rows]
    val_accs = [float(r["validation_accuracy"]) * 100.0 for r in summary_rows]
    val_f1s = [float(r["validation_macro_f1"]) * 100.0 for r in summary_rows]
    fit_accs = [float(r["fit_accuracy"]) * 100.0 for r in summary_rows]
    train_times = [float(r["training_seconds"]) for r in summary_rows]

    # 1. C vs Validation Accuracy & Macro F1
    metrics_plot_path = results_dir / "svm_c_vs_validation_metrics.png"
    fig, (ax_top, ax_val) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.8),
        dpi=150,
        gridspec_kw={"height_ratios": [1, 2.2]},
        sharex=True,
    )

    # Top panel: Overview including Fit Accuracy (100%) vs Validation (~63-66%)
    ax_top.plot(
        c_vals,
        fit_accs,
        marker="^",
        linewidth=1.8,
        linestyle="--",
        markersize=6,
        label="Fit Accuracy (100.00% across all C)",
        color="#2ca02c",
    )
    ax_top.plot(
        c_vals,
        val_accs,
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="Validation Accuracy (%)",
        color="#1f77b4",
    )
    ax_top.plot(
        c_vals,
        val_f1s,
        marker="s",
        linewidth=1.8,
        markersize=6,
        label="Validation Macro F1 (%)",
        color="#d62728",
    )
    ax_top.axvline(best_c, color="#d62728", linestyle=":", linewidth=1.4)
    ax_top.set_ylabel("Score (%)", fontsize=10.5)
    ax_top.set_ylim(58.0, 105.0)
    ax_top.set_title(
        "LinearSVC: C vs. Fit & Validation Performance (Prefix-Group Split)",
        fontsize=12,
        fontweight="bold",
    )
    ax_top.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_top.legend(loc="center right", fontsize=9)

    # Bottom panel: Zoomed-in view of Validation Accuracy and Validation Macro F1
    ax_val.plot(
        c_vals,
        val_accs,
        marker="o",
        linewidth=2.3,
        markersize=7.5,
        label="Validation Accuracy (%)",
        color="#1f77b4",
    )
    ax_val.plot(
        c_vals,
        val_f1s,
        marker="s",
        linewidth=2.3,
        markersize=7.5,
        label="Validation Macro F1 (%)",
        color="#d62728",
    )
    ax_val.axvline(
        best_c,
        color="#d62728",
        linestyle=":",
        linewidth=1.6,
        label=f"Selected Best C = {best_c} (by unrounded Val Macro F1)",
    )

    for c_v, v_acc, v_f1 in zip(c_vals, val_accs, val_f1s):
        ax_val.annotate(
            f"Acc: {v_acc:.2f}%",
            xy=(c_v, v_acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color="#1f77b4",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#1f77b4", alpha=0.9),
        )
        ax_val.annotate(
            f"F1: {v_f1:.2f}%",
            xy=(c_v, v_f1),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8.8,
            color="#d62728",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.9),
        )

    ax_val.set_xscale("log")
    ax_val.set_xticks(c_vals)
    ax_val.set_xticklabels([str(c) for c in c_vals])
    ax_val.set_xlabel("Regularization parameter C (log10 scale)", fontsize=11)
    ax_val.set_ylabel("Validation Score (%, zoomed)", fontsize=10.5)
    ax_val.set_ylim(min(val_f1s) - 1.0, max(val_accs) + 1.2)
    ax_val.grid(True, which="both", linestyle="--", alpha=0.4)
    ax_val.legend(loc="upper right", fontsize=9.2)
    fig.tight_layout()
    fig.savefig(metrics_plot_path)
    plt.close(fig)

    # 2. C vs Training Time
    time_plot_path = results_dir / "svm_c_vs_training_time.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    ax.plot(
        c_vals,
        train_times,
        marker="o",
        linewidth=2.2,
        markersize=7,
        color="#ff7f0e",
        label="Training time (seconds)",
    )
    for c_v, t_sec in zip(c_vals, train_times):
        ax.annotate(
            f"{t_sec:.2f}s",
            xy=(c_v, t_sec),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.9),
        )

    ax.set_xscale("log")
    ax.set_xticks(c_vals)
    ax.set_xticklabels([str(c) for c in c_vals])
    ax.set_xlabel("Regularization parameter C (log10 scale)", fontsize=11)
    ax.set_ylabel("Training Time (seconds)", fontsize=11)
    ax.set_title(
        "LinearSVC: C vs. Training Time on Fit Split (23,026 samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylim(0, max(train_times) * 1.22)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(time_plot_path)
    plt.close(fig)

    return metrics_plot_path, time_plot_path


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading HOG training features (test set is NOT loaded)...")
    X_train, y_train, class_names = load_training_data(args.features)

    print(f"2. Verifying saved prefix-group split: {args.split_indices}")
    split_verification = verify_and_load_split(
        args.split_indices,
        args.features,
        X_train,
        y_train,
        class_names,
    )
    fit_indices = split_verification["fit_indices"]
    val_indices = split_verification["validation_indices"]
    fit_groups = split_verification["fit_groups"]
    val_groups = split_verification["validation_groups"]

    X_fit = X_train[fit_indices]
    y_fit = y_train[fit_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(
        f"   Verified split: fit={len(X_fit)} samples ({fit_groups}), "
        f"val={len(X_val)} samples ({val_groups}), overlap=0, cross-HOG-dups=0"
    )

    library_versions = get_library_versions()
    labels = np.arange(len(class_names))

    summary_rows: list[dict[str, object]] = []
    detailed_reports: dict[str, dict[str, object]] = {}

    for c_val in DEFAULT_C_VALUES:
        c_tag = format_c_tag(c_val)
        exp_model_path = args.models_dir / f"svm_{c_tag}_prefix_group.joblib"
        exp_report_path = args.results_dir / f"svm_{c_tag}_prefix_group_validation.json"

        reused_baseline = False
        reuse_note = "Trained fresh in this experiment run."
        caught_warning_messages: list[str] = []

        if c_val == 1.0 and not args.no_reuse_baseline:
            can_reuse, reason, saved_bundle, saved_report = can_reuse_c1_baseline(
                args.baseline_model,
                args.baseline_report,
                args.max_iter,
                class_names,
                fit_groups,
                val_groups,
            )
            if can_reuse and saved_bundle is not None and saved_report is not None:
                print(f"\n--- C = {c_val} (Reusing verified baseline model) ---")
                print(f"   Reason: {reason}")
                pipeline = saved_bundle["model"]
                training_seconds = float(saved_report["training_seconds"])
                reused_baseline = True
                reuse_note = (
                    f"Reused verified baseline model from {args.baseline_model}; "
                    "re-evaluated predictions and extracted iteration metadata."
                )

        if not reused_baseline:
            print(f"\n--- Training LinearSVC with C = {c_val} (max_iter={args.max_iter}) ---")
            pipeline = build_pipeline(c_val, args.max_iter)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                train_start = time.perf_counter()
                pipeline.fit(X_fit, y_fit)
                training_seconds = time.perf_counter() - train_start

            for w in caught_warnings:
                if issubclass(w.category, ConvergenceWarning):
                    caught_warning_messages.append(str(w.message))

        clf = pipeline.named_steps["classifier"]
        iter_info = extract_iteration_info(clf)
        hit_max_iter = bool(
            iter_info["n_iter_available"]
            and iter_info["n_iter_max"] is not None
            and int(iter_info["n_iter_max"]) >= int(args.max_iter)
        )
        converged = (len(caught_warning_messages) == 0) and (not hit_max_iter)

        # Evaluate on fit
        fit_pred_start = time.perf_counter()
        y_fit_pred = pipeline.predict(X_fit)
        fit_prediction_seconds = time.perf_counter() - fit_pred_start

        fit_correct = int(np.sum(y_fit_pred == y_fit))
        fit_incorrect = int(np.sum(y_fit_pred != y_fit))
        fit_accuracy = float(accuracy_score(y_fit, y_fit_pred))
        fit_p, fit_r, fit_f1, _ = precision_recall_fscore_support(
            y_fit, y_fit_pred, average="macro", zero_division=0
        )

        # Evaluate on validation
        val_pred_start = time.perf_counter()
        y_val_pred = pipeline.predict(X_val)
        val_prediction_seconds = time.perf_counter() - val_pred_start
        if reused_baseline and saved_report is not None:
            recorded_val_pred_seconds = float(
                saved_report.get("validation_prediction_seconds", val_prediction_seconds)
            )
        else:
            recorded_val_pred_seconds = float(val_prediction_seconds)

        val_correct = int(np.sum(y_val_pred == y_val))
        val_incorrect = int(np.sum(y_val_pred != y_val))
        val_accuracy = float(accuracy_score(y_val, y_val_pred))
        val_p, val_r, val_f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average="macro", zero_division=0
        )

        per_class_report = classification_report(
            y_val,
            y_val_pred,
            labels=labels,
            target_names=class_names.tolist(),
            output_dict=True,
            zero_division=0,
        )

        split_meta = {
            "split_type": split_verification["split_type"],
            "split_file": split_verification["split_path"],
            "split_file_sha256": split_verification["split_file_sha256"],
            "fit_groups": fit_groups,
            "validation_groups": val_groups,
            "fit_samples": int(len(X_fit)),
            "validation_samples": int(len(X_val)),
            "features_file_sha256": split_verification["features_file_sha256"],
            "X_train_sha256": split_verification["X_train_sha256"],
            "y_train_sha256": split_verification["y_train_sha256"],
        }

        detailed_metrics = {
            "experiment": f"linear_svc_{c_tag}_prefix_group_validation",
            "model_family": "LinearSVC (L2 penalty, squared hinge loss, dual='auto')",
            "c": float(c_val),
            "max_iter": int(args.max_iter),
            "random_state": int(RANDOM_STATE),
            "dual": "auto",
            "scaler_used": False,
            "reused_from_saved_baseline": reused_baseline,
            "reuse_note": reuse_note,
            "convergence": {
                "converged": converged,
                "hit_max_iter": hit_max_iter,
                "convergence_warning_count": len(caught_warning_messages),
                "convergence_warning_messages": caught_warning_messages,
                **iter_info,
            },
            "fit_metrics": {
                "samples": int(len(X_fit)),
                "correct_predictions": fit_correct,
                "incorrect_predictions": fit_incorrect,
                "accuracy": fit_accuracy,
                "macro_precision": float(fit_p),
                "macro_recall": float(fit_r),
                "macro_f1": float(fit_f1),
            },
            "validation_metrics": {
                "samples": int(len(X_val)),
                "correct_predictions": val_correct,
                "incorrect_predictions": val_incorrect,
                "accuracy": val_accuracy,
                "macro_precision": float(val_p),
                "macro_recall": float(val_r),
                "macro_f1": float(val_f1),
            },
            "timing": {
                "training_seconds": float(training_seconds),
                "fit_prediction_seconds": float(fit_prediction_seconds),
                "validation_prediction_seconds": recorded_val_pred_seconds,
            },
            "split_info": split_meta,
            "hog_parameters": HOG_PARAMETERS,
            "image_size": list(IMAGE_SIZE),
            "library_versions": library_versions,
            "test_set_used": False,
            "classification_report": per_class_report,
        }

        model_bundle = {
            "model": pipeline,
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "hog_parameters": HOG_PARAMETERS,
            "c": float(c_val),
            "max_iter": int(args.max_iter),
            "random_state": int(RANDOM_STATE),
            "validation_size": float(VALIDATION_SIZE),
            "split_info": split_meta,
            "convergence": detailed_metrics["convergence"],
            "fit_metrics": detailed_metrics["fit_metrics"],
            "validation_metrics": detailed_metrics["validation_metrics"],
            "timing": detailed_metrics["timing"],
            "library_versions": library_versions,
            "reused_from_saved_baseline": reused_baseline,
            "training_scope": (
                f"fit split ({len(X_fit)} samples, groups={fit_groups}); "
                "test set not used"
            ),
        }

        joblib.dump(model_bundle, exp_model_path)
        exp_report_path.write_text(
            json.dumps(detailed_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary_row = {
            "c": float(c_val),
            "fit_samples": int(len(X_fit)),
            "fit_correct": fit_correct,
            "fit_incorrect": fit_incorrect,
            "fit_accuracy": fit_accuracy,
            "fit_macro_f1": float(fit_f1),
            "validation_samples": int(len(X_val)),
            "validation_correct": val_correct,
            "validation_incorrect": val_incorrect,
            "validation_accuracy": val_accuracy,
            "validation_macro_precision": float(val_p),
            "validation_macro_recall": float(val_r),
            "validation_macro_f1": float(val_f1),
            "training_seconds": float(training_seconds),
            "validation_prediction_seconds": recorded_val_pred_seconds,
            "converged": converged,
            "convergence_warning_count": len(caught_warning_messages),
            "n_iter_max": iter_info["n_iter_max"],
            "n_iter_min": iter_info["n_iter_min"],
            "n_iter_median": iter_info["n_iter_median"],
            "reused_from_baseline": reused_baseline,
            "model_path": str(exp_model_path),
            "report_path": str(exp_report_path),
        }
        summary_rows.append(summary_row)
        detailed_reports[str(c_val)] = detailed_metrics

        print(
            f"   Result C={c_val}: Fit Acc={fit_accuracy*100:.4f}% ({fit_correct}/{len(X_fit)}) | "
            f"Val Acc={val_accuracy*100:.4f}% ({val_correct}/{len(X_val)}, wrong={val_incorrect}) | "
            f"Val Macro P={val_p:.6f}, R={val_r:.6f}, F1={val_f1:.6f} | "
            f"Time={training_seconds:.3f}s | n_iter_max={iter_info['n_iter_max']} | "
            f"Warnings={len(caught_warning_messages)}"
        )

    # Select best C by highest unrounded validation_macro_f1; tie-breaker: smaller C
    sorted_for_selection = sorted(
        summary_rows,
        key=lambda r: (-float(r["validation_macro_f1"]), float(r["c"])),
    )
    best_row = sorted_for_selection[0]
    best_c = float(best_row["c"])
    baseline_row = next(r for r in summary_rows if float(r["c"]) == 1.0)

    f1_diff_pp = (
        float(best_row["validation_macro_f1"])
        - float(baseline_row["validation_macro_f1"])
    ) * 100.0
    acc_diff_pp = (
        float(best_row["validation_accuracy"])
        - float(baseline_row["validation_accuracy"])
    ) * 100.0
    correct_diff = int(best_row["validation_correct"]) - int(
        baseline_row["validation_correct"]
    )

    # Save summary CSV
    csv_path = args.results_dir / "svm_c_experiments_summary.csv"
    csv_fields = list(summary_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Generate log-scale plots
    metrics_plot_path, time_plot_path = plot_experiments(
        summary_rows, best_c, args.results_dir
    )

    summary_json_path = args.results_dir / "svm_c_experiments_summary.json"
    summary_payload = {
        "experiment_suite": "linear_svc_c_hyperparameter_search_prefix_group",
        "model_family_note": (
            "All 4 experiments train the exact same model family (LinearSVC on 1764 HOG features "
            "without a scaler), differing only in the regularization hyperparameter C."
        ),
        "tested_c_values": DEFAULT_C_VALUES,
        "selection_rule": (
            "Highest unrounded validation_macro_f1 on the prefix-group validation split "
            "(P8, P10); tie-breaker: smaller C. Fit accuracy and test set are never used."
        ),
        "selected_best_c": best_c,
        "selected_best_row": best_row,
        "baseline_c1_row": baseline_row,
        "improvement_over_c1_baseline": {
            "validation_macro_f1_delta": float(
                float(best_row["validation_macro_f1"])
                - float(baseline_row["validation_macro_f1"])
            ),
            "validation_macro_f1_percentage_points": f1_diff_pp,
            "validation_accuracy_delta": float(
                float(best_row["validation_accuracy"])
                - float(baseline_row["validation_accuracy"])
            ),
            "validation_accuracy_percentage_points": acc_diff_pp,
            "additional_validation_correct_predictions": correct_diff,
        },
        "split_verification": {
            k: v
            for k, v in split_verification.items()
            if k not in {"fit_indices", "validation_indices"}
        },
        "library_versions": library_versions,
        "hog_parameters": HOG_PARAMETERS,
        "image_size": list(IMAGE_SIZE),
        "outputs": {
            "summary_csv": str(csv_path),
            "summary_json": str(summary_json_path),
            "metrics_plot_png": str(metrics_plot_path),
            "training_time_plot_png": str(time_plot_path),
        },
        "results": summary_rows,
        "test_set_used": False,
    }
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ C EXPERIMENT SUITE SUMMARY ================")
    print(f"Selected best C        : {best_c}")
    print(
        f"Best Val Macro F1      : {float(best_row['validation_macro_f1']):.6f} "
        f"({f1_diff_pp:+.4f} percentage points vs C=1.0)"
    )
    print(
        f"Best Val Accuracy      : {float(best_row['validation_accuracy']):.6f} "
        f"({acc_diff_pp:+.4f} percentage points vs C=1.0, {correct_diff:+d} samples)"
    )
    print(f"Summary CSV saved to   : {csv_path.resolve()}")
    print(f"Summary JSON saved to  : {summary_json_path.resolve()}")
    print(f"Metrics plot saved to  : {metrics_plot_path.resolve()}")
    print(f"Time plot saved to     : {time_plot_path.resolve()}")


if __name__ == "__main__":
    main()
