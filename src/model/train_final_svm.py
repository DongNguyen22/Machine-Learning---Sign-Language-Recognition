"""Train the final LinearSVC(C=0.01) model on all 28,800 training samples and evaluate once on the 7,200 test samples.

Phase B pipeline:
1. Pre-training audit:
   - Verify data fingerprints against Phase A experiment summary (results/svm/svm_c_experiments_summary.json).
   - Confirm selected best C = 0.01 from prefix-group validation.
   - Validate X_train (28,800 x 1,764), y_train (28,800,), X_test (7,200 x 1,764), y_test (7,200,), and 36 classes.
   - Reconstruct and verify the 7,200 test image manifest against X_test and y_test, and audit prefix-group
     and SHA-256 / HOG overlap between train and test.
2. Final training:
   - Fit LinearSVC(C=0.01, random_state=42, max_iter=10000, dual='auto') inside a Pipeline (no scaler)
     on all 28,800 training samples.
   - Record training time, iteration counts (n_iter_), convergence status, warnings, and library versions.
   - Save final model bundle to models/svm/svm_final_c0_01.joblib (without overwriting any baseline or Phase A models).
3. Final evaluation (one-shot on X_test, plus full X_train for train/test gap analysis):
   - Compute Accuracy, Macro Precision, Macro Recall, Macro F1, and per-class metrics.
   - Measure prediction time on the 7,200 test HOG vectors (total and per-sample average), explicitly
     distinguishing prediction from pre-computed HOG vs. end-to-end raw image inference.
   - Save true and predicted labels (NPZ and CSV), confusion matrices (raw counts and row-normalized CSVs + PNGs),
     top confusion pairs CSV, per-class metrics CSV, and full JSON evaluation report.
   - Reload saved model to verify artifact integrity, prediction count (7,200), and confusion matrix sum (7,200).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
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
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_svm import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_FEATURE_COUNT,
    HOG_PARAMETERS,
    IMAGE_SIZE,
    RANDOM_STATE,
    build_pipeline,
)
from src.preprocess.build_hog_dataset import (
    IMAGE_EXTENSIONS,
    extract_hog,
    get_class_names,
)

FINAL_C = 0.01
EXPECTED_TRAIN_SAMPLES = 28_800
EXPECTED_TEST_SAMPLES = 7_200
EXPECTED_TRAIN_PER_CLASS = 800
EXPECTED_TEST_PER_CLASS = 200

DEFAULT_DATASET_PATH = Path(
    r"C:\Users\Quang Diep\Downloads\ASL-HG American Sign Language Hand Gesture Image D"
    r"\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000"
    r"\ASL_Processed_Images\asl_processed"
)
FILENAME_PATTERN = re.compile(r"^(P\d+)_([0-9A-Z])_(\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final LinearSVC(C=0.01) on 28,800 train samples and evaluate once on 7,200 test samples."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/svm_hog_features.npz"),
        help="Path to the HOG features NPZ file",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to asl_processed directory containing train/ and test/",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("results/svm/svm_train_image_manifest.csv"),
        help="Path to the verified training image manifest CSV",
    )
    parser.add_argument(
        "--test-manifest-output",
        type=Path,
        default=Path("results/svm/svm_test_image_manifest.csv"),
        help="Output path for the verified test image manifest CSV",
    )
    parser.add_argument(
        "--experiments-summary",
        type=Path,
        default=Path("results/svm/svm_c_experiments_summary.json"),
        help="Path to Phase A C-experiment summary JSON",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/svm/svm_final_c0_01.joblib"),
        help="Output path for the final trained model bundle",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/svm"),
        help="Directory to store all final evaluation outputs",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="Maximum iterations for LinearSVC (default: 10,000)",
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


def load_and_validate_full_dataset(
    features_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    with np.load(features_path, allow_pickle=False) as data:
        required_keys = {"X_train", "y_train", "X_test", "y_test", "class_names"}
        missing = required_keys.difference(data.files)
        if missing:
            raise ValueError(f"Feature file is missing keys: {sorted(missing)}")

        X_train = data["X_train"].copy()
        y_train = data["y_train"].copy()
        X_test = data["X_test"].copy()
        y_test = data["y_test"].copy()
        class_names = data["class_names"].astype(str).copy()

    # Validate shapes, dtypes, finite values, and class distributions
    if X_train.shape != (EXPECTED_TRAIN_SAMPLES, EXPECTED_FEATURE_COUNT):
        raise ValueError(
            f"Expected X_train shape {(EXPECTED_TRAIN_SAMPLES, EXPECTED_FEATURE_COUNT)}, got {X_train.shape}"
        )
    if y_train.shape != (EXPECTED_TRAIN_SAMPLES,):
        raise ValueError(f"Expected y_train shape {(EXPECTED_TRAIN_SAMPLES,)}, got {y_train.shape}")
    if X_test.shape != (EXPECTED_TEST_SAMPLES, EXPECTED_FEATURE_COUNT):
        raise ValueError(
            f"Expected X_test shape {(EXPECTED_TEST_SAMPLES, EXPECTED_FEATURE_COUNT)}, got {X_test.shape}"
        )
    if y_test.shape != (EXPECTED_TEST_SAMPLES,):
        raise ValueError(f"Expected y_test shape {(EXPECTED_TEST_SAMPLES,)}, got {y_test.shape}")
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"Expected {EXPECTED_CLASS_COUNT} class_names, got {len(class_names)}")

    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values")
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinite values")

    expected_labels = np.arange(EXPECTED_CLASS_COUNT, dtype=np.int32)
    if not np.array_equal(np.unique(y_train), expected_labels):
        raise ValueError("y_train does not contain all 36 class labels 0..35")
    if not np.array_equal(np.unique(y_test), expected_labels):
        raise ValueError("y_test does not contain all 36 class labels 0..35")

    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())
    if set(train_counts.values()) != {EXPECTED_TRAIN_PER_CLASS}:
        raise ValueError(f"Train class counts are not uniform at 800: {train_counts}")
    if set(test_counts.values()) != {EXPECTED_TEST_PER_CLASS}:
        raise ValueError(f"Test class counts are not uniform at 200: {test_counts}")

    fingerprints = {
        "features_file": str(features_path),
        "features_file_size_bytes": int(features_path.stat().st_size),
        "features_file_sha256": sha256_file(features_path),
        "X_train_sha256": sha256_array(X_train),
        "y_train_sha256": sha256_array(y_train),
        "X_test_sha256": sha256_array(X_test),
        "y_test_sha256": sha256_array(y_test),
        "class_names_sha256": sha256_array(np.asarray(class_names, dtype="U1")),
    }
    return X_train, y_train, X_test, y_test, class_names, fingerprints


def build_and_audit_test_manifest(
    dataset_path: Path,
    train_manifest_path: Path,
    test_manifest_output: Path,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Load verified train manifest
    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")

    train_rows: list[dict[str, object]] = []
    train_sha_to_rows: dict[str, list[int]] = defaultdict(list)
    train_filenames: set[str] = set()
    train_prefixes: Counter[str] = Counter()
    train_seq_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)

    with train_manifest_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = int(r["row_index"])
            sha = r["file_sha256"]
            fname = r["filename"]
            pfx = r["prefix_group"]
            cname = r["class_name"]
            seq = int(r["sequence_number"])
            train_rows.append(r)
            train_sha_to_rows[sha].append(idx)
            train_filenames.add(fname)
            train_prefixes[pfx] += 1
            train_seq_by_group[(pfx, cname)].add(seq)

    test_dir = dataset_path / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {test_dir}")

    fs_class_names = get_class_names(test_dir)
    if fs_class_names != class_names.tolist():
        raise ValueError("Test folder class names do not match class_names in NPZ")

    test_rows: list[dict[str, object]] = []
    block_boundaries: dict[tuple[str, int], list[int]] = {}
    row_idx = 0

    for class_index, class_name in enumerate(fs_class_names):
        class_dir = test_dir / class_name
        image_paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for p in image_paths:
            match = FILENAME_PATTERN.match(p.name)
            if not match:
                raise ValueError(f"Unexpected test filename format: {p.name}")
            prefix_group = match.group(1).upper()
            label_token = match.group(2).upper()
            seq_num = int(match.group(3))
            if label_token != class_name.upper():
                raise ValueError(
                    f"Filename label {label_token} != folder {class_name} for {p}"
                )
            file_hash = sha256_file(p)
            rel_path = p.relative_to(dataset_path).as_posix()
            test_rows.append(
                {
                    "row_index": row_idx,
                    "relative_path": rel_path,
                    "filename": p.name,
                    "prefix_group": prefix_group,
                    "class_name": class_name,
                    "class_index": class_index,
                    "sequence_number": seq_num,
                    "file_sha256": file_hash,
                    "abs_path": p,
                }
            )
            key = (prefix_group, class_index)
            if key not in block_boundaries:
                block_boundaries[key] = [row_idx, row_idx]
            else:
                block_boundaries[key][1] = row_idx
            row_idx += 1

    if len(test_rows) != len(X_test):
        raise ValueError(f"Test manifest count {len(test_rows)} != X_test count {len(X_test)}")

    reconstructed_y_test = np.array([int(r["class_index"]) for r in test_rows], dtype=np.int32)
    if not np.array_equal(reconstructed_y_test, y_test):
        raise ValueError("Reconstructed test label order does not match y_test")

    # Verify HOG extraction alignment at start and end of every single (prefix_group, class) block in test (720 images)
    indices_to_verify: list[int] = []
    for first_idx, last_idx in block_boundaries.values():
        indices_to_verify.append(first_idx)
        if last_idx != first_idx:
            indices_to_verify.append(last_idx)

    verified_hog_count = 0
    for idx in indices_to_verify:
        img_path = Path(test_rows[idx]["abs_path"])
        hog_vec = extract_hog(img_path)
        if hog_vec is None or not np.array_equal(hog_vec, X_test[idx]):
            raise ValueError(f"Test HOG mismatch at row_index={idx}, file={img_path}")
        verified_hog_count += 1

    # Write test manifest CSV
    test_manifest_output.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "row_index",
        "relative_path",
        "filename",
        "prefix_group",
        "class_name",
        "class_index",
        "sequence_number",
        "file_sha256",
    ]
    with test_manifest_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in test_rows:
            writer.writerow({k: r[k] for k in csv_fields})

    # Audit overlap between test (7,200) and train (28,800)
    filename_overlap = sum(1 for r in test_rows if str(r["filename"]) in train_filenames)

    train_hog_bytes = {X_train[i].tobytes() for i in range(len(X_train))}
    test_exact_sha_in_train_indices: list[int] = []
    test_exact_hog_in_train_indices: list[int] = []
    exact_duplicate_examples: list[dict[str, object]] = []

    test_prefixes: Counter[str] = Counter()
    test_with_adjacent_seq_in_train = 0

    for idx, r in enumerate(test_rows):
        sha = str(r["file_sha256"])
        pfx = str(r["prefix_group"])
        cname = str(r["class_name"])
        seq = int(r["sequence_number"])
        test_prefixes[pfx] += 1

        if (seq - 1 in train_seq_by_group[(pfx, cname)]) or (
            seq + 1 in train_seq_by_group[(pfx, cname)]
        ):
            test_with_adjacent_seq_in_train += 1

        if sha in train_sha_to_rows:
            test_exact_sha_in_train_indices.append(idx)
            if len(exact_duplicate_examples) < 10:
                matching_train_idxs = train_sha_to_rows[sha]
                exact_duplicate_examples.append(
                    {
                        "test_row_index": idx,
                        "test_filename": str(r["filename"]),
                        "matching_train_row_indices": matching_train_idxs,
                        "matching_train_filenames": [
                            str(train_rows[m]["filename"]) for m in matching_train_idxs
                        ],
                        "file_sha256": sha,
                    }
                )

        if X_test[idx].tobytes() in train_hog_bytes:
            test_exact_hog_in_train_indices.append(idx)

    shared_prefix_groups = sorted(
        set(train_prefixes.keys()) & set(test_prefixes.keys()),
        key=lambda x: int(x[1:]),
    )
    shared_blocks_count = len(set(train_seq_by_group.keys()) & {(str(r["prefix_group"]), str(r["class_name"])) for r in test_rows})

    audit_summary = {
        "test_manifest_csv": str(test_manifest_output),
        "test_manifest_sha256": sha256_file(test_manifest_output),
        "test_rows_verified": len(test_rows),
        "y_test_exact_match": True,
        "test_prefix_class_blocks": len(block_boundaries),
        "test_boundary_hog_vectors_verified": verified_hog_count,
        "test_boundary_hog_bitwise_match_ratio": 1.0,
        "exact_filename_overlap_train_vs_test": filename_overlap,
        "shared_prefix_groups_between_train_and_test": shared_prefix_groups,
        "shared_prefix_group_count": len(shared_prefix_groups),
        "shared_prefix_class_blocks_between_train_and_test": shared_blocks_count,
        "test_samples_by_prefix": {
            k: test_prefixes[k] for k in sorted(test_prefixes.keys(), key=lambda x: int(x[1:]))
        },
        "test_samples_with_adjacent_seq_num_pm1_in_train": test_with_adjacent_seq_in_train,
        "test_samples_with_adjacent_seq_num_pm1_ratio": float(
            test_with_adjacent_seq_in_train / len(X_test)
        ),
        "exact_file_sha256_duplicates_in_train_count": len(test_exact_sha_in_train_indices),
        "exact_file_sha256_duplicates_in_train_ratio": float(
            len(test_exact_sha_in_train_indices) / len(X_test)
        ),
        "exact_hog_duplicates_in_train_count": len(test_exact_hog_in_train_indices),
        "exact_hog_duplicates_in_train_ratio": float(
            len(test_exact_hog_in_train_indices) / len(X_test)
        ),
        "test_exact_sha_in_train_indices": test_exact_sha_in_train_indices,
        "example_train_test_exact_file_duplicates": exact_duplicate_examples,
        "provenance_finding": (
            "The original asl_processed dataset split each of the 360 (P_k, class) 100-image series "
            "80/20 into train (80 images) and test (20 images) by interleaved sequence numbers. "
            "Consequently, all 10 filename prefix groups (P1..P10) and all 360 (P_k, class) blocks "
            "appear in both train and test, 86.26% of test images have an adjacent sequence number "
            "(seq +/- 1) in train, and 165 test images (2.29%) have byte-identical SHA-256 files "
            "(under different filenames) in train."
        ),
    }
    return test_rows, audit_summary


def save_confusion_matrix_csv(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    float_format: str | None = None,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label\\predicted_label", *class_names])
        for row_idx, cname in enumerate(class_names):
            if float_format is None:
                row_vals = [int(v) for v in cm[row_idx]]
            else:
                row_vals = [format(float(v), float_format) for v in cm[row_idx]]
            writer.writerow([cname, *row_vals])


def plot_confusion_matrices(
    cm_counts: np.ndarray,
    cm_norm: np.ndarray,
    class_names: list[str],
    counts_png_path: Path,
    norm_png_path: Path,
) -> None:
    n_classes = len(class_names)
    ticks = np.arange(n_classes)

    # 1. Raw counts confusion matrix
    fig, ax = plt.subplots(figsize=(14.5, 13.0), dpi=150)
    im = ax.imshow(cm_counts, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of Test Samples", fontsize=11)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Sign Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Sign Label", fontsize=12, fontweight="bold")
    ax.set_title(
        "Final LinearSVC (C=0.01) Confusion Matrix — Raw Sample Counts (Test Set: 7,200 samples)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    thresh = cm_counts.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = int(cm_counts[i, j])
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "#111111",
                    fontsize=6.8 if val >= 100 else 7.2,
                    fontweight="bold" if i == j or val >= 3 else "normal",
                )

    fig.tight_layout()
    fig.savefig(counts_png_path)
    plt.close(fig)

    # 2. Row-normalized confusion matrix (by true label)
    fig, ax = plt.subplots(figsize=(14.5, 13.0), dpi=150)
    im = ax.imshow(cm_norm * 100.0, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage of True Class (%)", fontsize=11)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted Sign Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Sign Label", fontsize=12, fontweight="bold")
    ax.set_title(
        "Final LinearSVC (C=0.01) Confusion Matrix — Normalized by True Label (%)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    for i in range(n_classes):
        for j in range(n_classes):
            pct = float(cm_norm[i, j]) * 100.0
            if pct >= 0.1:
                txt = "100" if pct >= 99.99 else f"{pct:.1f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if pct > 50.0 else "#111111",
                    fontsize=6.5,
                    fontweight="bold" if i == j or pct >= 0.5 else "normal",
                )

    fig.tight_layout()
    fig.savefig(norm_png_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("1. Pre-training check: loading dataset and verifying fingerprints...")
    X_train, y_train, X_test, y_test, class_names, fingerprints = (
        load_and_validate_full_dataset(args.features)
    )

    if not args.experiments_summary.is_file():
        raise FileNotFoundError(
            f"Phase A experiments summary not found: {args.experiments_summary}"
        )
    exp_summary = json.loads(args.experiments_summary.read_text(encoding="utf-8"))
    selected_c = float(exp_summary["selected_best_c"])
    if selected_c != FINAL_C:
        raise ValueError(
            f"Expected selected_best_c={FINAL_C} from Phase A, got {selected_c}"
        )

    split_ver = exp_summary.get("split_verification", {})
    if (
        split_ver.get("features_file_sha256") != fingerprints["features_file_sha256"]
        or split_ver.get("X_train_sha256") != fingerprints["X_train_sha256"]
        or split_ver.get("y_train_sha256") != fingerprints["y_train_sha256"]
    ):
        raise ValueError("Dataset SHA-256 fingerprints do not match Phase A experiment records!")

    print(
        f"   Verified dataset shapes: X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}, classes={len(class_names)}"
    )
    print(f"   Confirmed selected C={FINAL_C} from Phase A prefix-group validation.")

    print("2. Auditing test manifest and train/test provenance & duplicate overlap...")
    test_rows, audit_summary = build_and_audit_test_manifest(
        args.dataset_path,
        args.train_manifest,
        args.test_manifest_output,
        X_train,
        X_test,
        y_test,
        class_names,
    )
    print(
        f"   Verified test manifest: {audit_summary['test_rows_verified']} rows, "
        f"boundary HOG check={audit_summary['test_boundary_hog_vectors_verified']}/720, "
        f"filename overlap={audit_summary['exact_filename_overlap_train_vs_test']}, "
        f"shared prefix groups={audit_summary['shared_prefix_group_count']}/10, "
        f"exact SHA-256/HOG duplicates with train={audit_summary['exact_file_sha256_duplicates_in_train_count']}/7200"
    )

    # 3. Train final LinearSVC(C=0.01) on all 28,800 training samples
    print(
        f"\n3. Training final LinearSVC(C={FINAL_C}, max_iter={args.max_iter}, random_state={RANDOM_STATE}) "
        f"on all {len(X_train)} training samples..."
    )
    pipeline = build_pipeline(FINAL_C, args.max_iter)
    caught_warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        train_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
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

    print(
        f"   Finished training in {training_seconds:.3f}s | converged={converged} | "
        f"n_iter_max={iter_info['n_iter_max']} (min={iter_info['n_iter_min']}, "
        f"median={iter_info['n_iter_median']}) | warnings={len(caught_warning_messages)}"
    )

    # 4. Evaluate on full X_train (28,800) to measure train/test gap
    train_pred_start = time.perf_counter()
    y_train_pred = pipeline.predict(X_train)
    train_prediction_seconds = time.perf_counter() - train_pred_start

    train_correct = int(np.sum(y_train_pred == y_train))
    train_incorrect = int(np.sum(y_train_pred != y_train))
    train_accuracy = float(accuracy_score(y_train, y_train_pred))
    train_p, train_r, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="macro", zero_division=0
    )

    # 5. Evaluate ONCE on full X_test (7,200)
    print(f"\n4. Evaluating final model on all {len(X_test)} test samples (from pre-extracted HOG features)...")
    test_pred_start = time.perf_counter()
    y_test_pred = pipeline.predict(X_test)
    test_prediction_seconds = time.perf_counter() - test_pred_start
    test_prediction_ms_per_sample = (test_prediction_seconds / len(X_test)) * 1000.0

    test_decision_scores = pipeline.decision_function(X_test)

    test_correct = int(np.sum(y_test_pred == y_test))
    test_incorrect = int(np.sum(y_test_pred != y_test))
    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="macro", zero_division=0
    )

    # Also compute supplementary metrics on the 7,035 deduplicated test samples
    # (excluding the 165 test samples that have an exact SHA-256/HOG duplicate in train)
    dup_set = set(audit_summary["test_exact_sha_in_train_indices"])
    non_dup_mask = np.array([i not in dup_set for i in range(len(X_test))], dtype=bool)
    y_test_dedup = y_test[non_dup_mask]
    y_pred_dedup = y_test_pred[non_dup_mask]
    dedup_correct = int(np.sum(y_pred_dedup == y_test_dedup))
    dedup_incorrect = int(np.sum(y_pred_dedup != y_test_dedup))
    dedup_acc = float(accuracy_score(y_test_dedup, y_pred_dedup))
    dedup_p, dedup_r, dedup_f1, _ = precision_recall_fscore_support(
        y_test_dedup, y_pred_dedup, average="macro", zero_division=0
    )

    labels = np.arange(len(class_names))
    class_list = class_names.tolist()
    report_dict = classification_report(
        y_test,
        y_test_pred,
        labels=labels,
        target_names=class_list,
        output_dict=True,
        zero_division=0,
    )

    # Confusion matrices
    cm_counts = confusion_matrix(y_test, y_test_pred, labels=labels)
    cm_norm = confusion_matrix(y_test, y_test_pred, labels=labels, normalize="true")

    if int(np.sum(cm_counts)) != len(X_test):
        raise ValueError(
            f"Confusion matrix sum {int(np.sum(cm_counts))} != test size {len(X_test)}"
        )

    # Extract per-class metrics table
    per_class_csv_path = args.results_dir / "svm_final_c0_01_per_class_metrics.csv"
    per_class_rows: list[dict[str, object]] = []
    for c_idx, cname in enumerate(class_list):
        c_stats = report_dict[cname]
        support = int(cm_counts[c_idx].sum())
        correct_c = int(cm_counts[c_idx, c_idx])
        incorrect_c = support - correct_c
        per_class_rows.append(
            {
                "class_index": c_idx,
                "class_name": cname,
                "support": support,
                "correct_predictions": correct_c,
                "incorrect_predictions": incorrect_c,
                "precision": float(c_stats["precision"]),
                "recall": float(c_stats["recall"]),
                "f1_score": float(c_stats["f1-score"]),
            }
        )

    with per_class_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()))
        writer.writeheader()
        for r in per_class_rows:
            writer.writerow(r)

    # Extract all non-zero off-diagonal confusion pairs sorted by count descending
    confusion_pairs: list[dict[str, object]] = []
    for i in range(len(class_list)):
        row_support = int(cm_counts[i].sum())
        for j in range(len(class_list)):
            if i != j and cm_counts[i, j] > 0:
                cnt = int(cm_counts[i, j])
                confusion_pairs.append(
                    {
                        "true_class_index": i,
                        "true_class": class_list[i],
                        "predicted_class_index": j,
                        "predicted_class": class_list[j],
                        "pair_label": f"{class_list[i]} -> {class_list[j]}",
                        "count": cnt,
                        "true_class_support": row_support,
                        "rate_within_true_class": float(cnt / row_support),
                    }
                )
    confusion_pairs.sort(
        key=lambda x: (-int(x["count"]), str(x["true_class"]), str(x["predicted_class"]))
    )

    top_confusions_csv_path = args.results_dir / "svm_final_c0_01_top_confusions.csv"
    with top_confusions_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "pair_label",
            "true_class",
            "predicted_class",
            "count",
            "true_class_support",
            "rate_within_true_class",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if confusion_pairs:
            for rank, item in enumerate(confusion_pairs, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        "pair_label": item["pair_label"],
                        "true_class": item["true_class"],
                        "predicted_class": item["predicted_class"],
                        "count": item["count"],
                        "true_class_support": item["true_class_support"],
                        "rate_within_true_class": item["rate_within_true_class"],
                    }
                )

    # Save confusion matrices (CSV + PNG)
    cm_counts_csv_path = args.results_dir / "svm_final_c0_01_confusion_matrix_counts.csv"
    cm_norm_csv_path = args.results_dir / "svm_final_c0_01_confusion_matrix_normalized.csv"
    cm_counts_png_path = args.results_dir / "svm_final_c0_01_confusion_matrix_counts.png"
    cm_norm_png_path = args.results_dir / "svm_final_c0_01_confusion_matrix_normalized.png"

    save_confusion_matrix_csv(cm_counts, class_list, cm_counts_csv_path, float_format=None)
    save_confusion_matrix_csv(cm_norm, class_list, cm_norm_csv_path, float_format=".6f")
    plot_confusion_matrices(
        cm_counts, cm_norm, class_list, cm_counts_png_path, cm_norm_png_path
    )

    # Save true and predicted labels (NPZ + CSV)
    predictions_npz_path = args.results_dir / "svm_final_c0_01_test_predictions.npz"
    predictions_csv_path = args.results_dir / "svm_final_c0_01_test_predictions.csv"

    test_filenames_arr = np.array([str(r["filename"]) for r in test_rows])
    test_prefixes_arr = np.array([str(r["prefix_group"]) for r in test_rows])

    np.savez_compressed(
        predictions_npz_path,
        y_true=y_test,
        y_pred=y_test_pred,
        true_labels=class_names[y_test],
        predicted_labels=class_names[y_test_pred],
        correct=(y_test_pred == y_test),
        decision_scores=test_decision_scores.astype(np.float32),
        filenames=test_filenames_arr,
        prefix_groups=test_prefixes_arr,
        class_names=class_names,
    )

    top2_sorted = np.sort(test_decision_scores, axis=1)
    top1_minus_top2_margin = top2_sorted[:, -1] - top2_sorted[:, -2]
    pred_top1_scores = test_decision_scores[np.arange(len(y_test)), y_test_pred]

    with predictions_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_index",
                "filename",
                "prefix_group",
                "y_true_index",
                "true_label",
                "y_pred_index",
                "predicted_label",
                "is_correct",
                "predicted_decision_score",
                "margin_top1_minus_top2",
                "has_exact_sha256_duplicate_in_train",
            ]
        )
        for idx in range(len(y_test)):
            writer.writerow(
                [
                    idx,
                    test_rows[idx]["filename"],
                    test_rows[idx]["prefix_group"],
                    int(y_test[idx]),
                    class_list[int(y_test[idx])],
                    int(y_test_pred[idx]),
                    class_list[int(y_test_pred[idx])],
                    bool(y_test_pred[idx] == y_test[idx]),
                    f"{float(pred_top1_scores[idx]):.6f}",
                    f"{float(top1_minus_top2_margin[idx]):.6f}",
                    bool(idx in dup_set),
                ]
            )

    library_versions = get_library_versions()

    # Save final model bundle
    model_bundle = {
        "model": pipeline,
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "hog_parameters": HOG_PARAMETERS,
        "c": float(FINAL_C),
        "max_iter": int(args.max_iter),
        "random_state": int(RANDOM_STATE),
        "dual": "auto",
        "scaler_used": False,
        "training_scope": "Full X_train (28,800 samples across all 36 classes and P1..P10)",
        "selected_via": {
            "validation_split_type": "prefix_group_shuffle_split (fit: P1..P7,P9; val: P8,P10)",
            "tested_c_values": [0.01, 0.1, 1.0, 10.0],
            "selected_c": float(FINAL_C),
            "validation_macro_f1": float(
                exp_summary["selected_best_row"]["validation_macro_f1"]
            ),
            "validation_accuracy": float(
                exp_summary["selected_best_row"]["validation_accuracy"]
            ),
        },
        "convergence": {
            "converged": converged,
            "hit_max_iter": hit_max_iter,
            "convergence_warning_count": len(caught_warning_messages),
            "convergence_warning_messages": caught_warning_messages,
            **iter_info,
        },
        "train_metrics": {
            "samples": int(len(X_train)),
            "correct_predictions": train_correct,
            "incorrect_predictions": train_incorrect,
            "accuracy": train_accuracy,
            "macro_precision": float(train_p),
            "macro_recall": float(train_r),
            "macro_f1": float(train_f1),
        },
        "test_metrics": {
            "samples": int(len(X_test)),
            "correct_predictions": test_correct,
            "incorrect_predictions": test_incorrect,
            "accuracy": test_accuracy,
            "macro_precision": float(test_p),
            "macro_recall": float(test_r),
            "macro_f1": float(test_f1),
        },
        "timing": {
            "training_seconds": float(training_seconds),
            "train_prediction_seconds": float(train_prediction_seconds),
            "test_prediction_seconds_total": float(test_prediction_seconds),
            "test_prediction_ms_per_sample": float(test_prediction_ms_per_sample),
            "timing_scope_note": (
                "Prediction timing measures LinearSVC inference from pre-extracted "
                "1764-dim HOG vectors in RAM (excludes image disk read, resize/padding, "
                "and HOG feature extraction)."
            ),
        },
        "data_fingerprints": fingerprints,
        "library_versions": library_versions,
    }
    joblib.dump(model_bundle, args.model_output)

    # 6. Reload saved model and verify integrity
    reloaded_bundle = joblib.load(args.model_output)
    reloaded_pipeline = reloaded_bundle["model"]
    reloaded_preds = reloaded_pipeline.predict(X_test)
    model_reload_verified = bool(
        len(reloaded_preds) == len(X_test)
        and np.array_equal(reloaded_preds, y_test_pred)
        and int(np.sum(cm_counts)) == len(X_test)
    )
    if not model_reload_verified:
        raise RuntimeError("Reloaded model verification failed!")

    # Save full evaluation JSON
    eval_json_path = args.results_dir / "svm_final_c0_01_test_evaluation.json"
    evaluation_report = {
        "experiment": "linear_svc_final_c0_01_full_train_test_evaluation",
        "model_family": "LinearSVC (L2 penalty, squared hinge loss, dual='auto', no scaler)",
        "hyperparameters": {
            "C": float(FINAL_C),
            "max_iter": int(args.max_iter),
            "random_state": int(RANDOM_STATE),
            "dual": "auto",
            "scaler_used": False,
        },
        "hog_parameters": HOG_PARAMETERS,
        "image_size": list(IMAGE_SIZE),
        "hyperparameter_selection_provenance": {
            "note": (
                "C=0.01 was selected strictly on the prefix-group validation split "
                "(Fit: P1..P7, P9 = 23,026 samples; Validation: P8, P10 = 5,774 samples) "
                "before any test set evaluation. Differences between prefix-group validation "
                "metrics (~65.47% accuracy) and final test metrics reflect the different split "
                "structures of the two evaluation sets (unseen prefix groups vs. shared prefix groups), "
                "NOT a model change."
            ),
            "prefix_group_validation_accuracy": float(
                exp_summary["selected_best_row"]["validation_accuracy"]
            ),
            "prefix_group_validation_macro_precision": float(
                exp_summary["selected_best_row"]["validation_macro_precision"]
            ),
            "prefix_group_validation_macro_recall": float(
                exp_summary["selected_best_row"]["validation_macro_recall"]
            ),
            "prefix_group_validation_macro_f1": float(
                exp_summary["selected_best_row"]["validation_macro_f1"]
            ),
        },
        "convergence": model_bundle["convergence"],
        "full_train_metrics": model_bundle["train_metrics"],
        "final_test_metrics_all_7200": {
            "samples": int(len(X_test)),
            "correct_predictions": test_correct,
            "incorrect_predictions": test_incorrect,
            "accuracy": test_accuracy,
            "macro_precision": float(test_p),
            "macro_recall": float(test_r),
            "macro_f1": float(test_f1),
            "decision_margin_top1_minus_top2": {
                "min": float(np.min(top1_minus_top2_margin)),
                "p01": float(np.percentile(top1_minus_top2_margin, 1)),
                "median": float(np.median(top1_minus_top2_margin)),
                "mean": float(np.mean(top1_minus_top2_margin)),
                "max": float(np.max(top1_minus_top2_margin)),
            },
        },
        "supplementary_test_metrics_excluding_165_exact_train_duplicates": {
            "samples": int(len(y_test_dedup)),
            "excluded_exact_sha256_duplicate_samples": int(len(dup_set)),
            "correct_predictions": dedup_correct,
            "incorrect_predictions": dedup_incorrect,
            "accuracy": dedup_acc,
            "macro_precision": float(dedup_p),
            "macro_recall": float(dedup_r),
            "macro_f1": float(dedup_f1),
        },
        "train_test_gap": {
            "train_minus_test_accuracy": float(train_accuracy - test_accuracy),
            "train_minus_test_macro_f1": float(float(train_f1) - float(test_f1)),
        },
        "timing": model_bundle["timing"],
        "confusion_analysis": {
            "total_misclassified_test_samples": test_incorrect,
            "distinct_confused_class_pairs_count": len(confusion_pairs),
            "top_20_confusion_pairs": confusion_pairs[:20],
        },
        "train_test_provenance_and_overlap_audit": {
            k: v
            for k, v in audit_summary.items()
            if k != "test_exact_sha_in_train_indices"
        },
        "artifact_verification": {
            "model_reloaded_successfully": model_reload_verified,
            "prediction_count_matches_test_size_7200": int(len(reloaded_preds)) == len(X_test),
            "confusion_matrix_sum_matches_7200": int(np.sum(cm_counts)) == len(X_test),
        },
        "saved_files": {
            "final_model_joblib": str(args.model_output),
            "evaluation_json": str(eval_json_path),
            "per_class_metrics_csv": str(per_class_csv_path),
            "top_confusions_csv": str(top_confusions_csv_path),
            "test_predictions_npz": str(predictions_npz_path),
            "test_predictions_csv": str(predictions_csv_path),
            "confusion_matrix_counts_csv": str(cm_counts_csv_path),
            "confusion_matrix_normalized_csv": str(cm_norm_csv_path),
            "confusion_matrix_counts_png": str(cm_counts_png_path),
            "confusion_matrix_normalized_png": str(cm_norm_png_path),
            "test_manifest_csv": str(args.test_manifest_output),
        },
        "data_fingerprints": fingerprints,
        "library_versions": library_versions,
        "classification_report": report_dict,
    }
    eval_json_path.write_text(
        json.dumps(evaluation_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ FINAL SVM (C=0.01) EVALUATION SUMMARY ================")
    print(
        f"Full Train (28,800) : Acc={train_accuracy*100:.4f}% ({train_correct}/{len(X_train)}, wrong={train_incorrect}) | "
        f"Macro F1={train_f1:.6f}"
    )
    print(
        f"Final Test (7,200)  : Acc={test_accuracy*100:.4f}% ({test_correct}/{len(X_test)}, wrong={test_incorrect}) | "
        f"Macro P={test_p:.6f}, R={test_r:.6f}, F1={test_f1:.6f}"
    )
    print(
        f"Dedup Test (7,035)  : Acc={dedup_acc*100:.4f}% ({dedup_correct}/{len(y_test_dedup)}, wrong={dedup_incorrect}) | "
        f"Macro F1={dedup_f1:.6f}"
    )
    print(
        f"Timing (HOG -> pred): Total={test_prediction_seconds:.4f}s for 7,200 samples "
        f"({test_prediction_ms_per_sample:.4f} ms/sample)"
    )
    print(f"Top confusion pairs : {confusion_pairs[:10]}")
    print(f"Artifact verification: reload={model_reload_verified}, cm_sum={int(np.sum(cm_counts))}")


if __name__ == "__main__":
    main()
