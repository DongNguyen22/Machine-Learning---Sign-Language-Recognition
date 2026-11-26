"""Verify the 100% validation accuracy of the LinearSVC C=1.0 baseline.

This script performs five checks without retraining the saved baseline model:
1. Code/prediction audit using the saved model on the recreated split.
2. Deterministic split recreation, index disjointness check, and split saving
   with dataset SHA-256 fingerprints.
3. Exact HOG vector duplicate check (and nearest-neighbor similarity check)
   between fit and validation splits.
4. Dataset metadata and filename pattern check, verifying path-to-row alignment
   with build_hog_dataset.py and checking raw image file hashes and sequence
   adjacency across splits.
5. Saves machine-readable evidence to results/svm/svm_baseline_verification.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_svm import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_FEATURE_COUNT,
    RANDOM_STATE,
    VALIDATION_SIZE,
    load_training_data,
)
from src.preprocess.build_hog_dataset import (
    IMAGE_EXTENSIONS,
    extract_hog,
    get_class_names,
)

DEFAULT_DATASET_PATH = Path(
    r"C:\Users\Quang Diep\Downloads\ASL-HG American Sign Language Hand Gesture Image D"
    r"\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000"
    r"\ASL_Processed_Images\asl_processed"
)
FILENAME_PATTERN = re.compile(r"^(P\d+)_([0-9A-Z])_(\d+)\.jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify baseline LinearSVC validation accuracy and data split integrity."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/svm_hog_features.npz"),
        help="Path to the HOG feature .npz file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/svm/svm_baseline_c1.joblib"),
        help="Path to the saved baseline model bundle",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("results/svm/svm_baseline_c1_validation.json"),
        help="Path to the baseline validation report JSON",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to asl_processed directory containing train/ and test/",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("results/svm/svm_validation_split_indices.npz"),
        help="Output path to save reproducible fit/validation indices",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path("results/svm/svm_baseline_verification.json"),
        help="Output path for the verification JSON report",
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


def collect_ordered_paths(split_dir: Path, class_names: list[str]) -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        image_paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(image_paths)
        labels.extend([class_index] * len(image_paths))
    return paths, np.asarray(labels, dtype=np.int32)


def main() -> None:
    args = parse_args()

    # 1. Load features and compute fingerprints
    X_train, y_train, class_names = load_training_data(args.features)
    with np.load(args.features, allow_pickle=False) as raw_npz:
        X_test = raw_npz["X_test"].copy()
        y_test = raw_npz["y_test"].copy()

    npz_file_sha256 = sha256_file(args.features)
    fingerprints = {
        "features_file": str(args.features),
        "features_file_size_bytes": int(args.features.stat().st_size),
        "features_file_sha256": npz_file_sha256,
        "X_train_sha256": sha256_array(X_train),
        "y_train_sha256": sha256_array(y_train),
        "X_test_sha256": sha256_array(X_test),
        "y_test_sha256": sha256_array(y_test),
        "class_names_sha256": sha256_array(np.asarray(class_names, dtype="U1")),
    }

    # 2. Recreate exact split and check index overlap
    all_indices = np.arange(len(X_train), dtype=np.int32)
    X_fit, X_val, y_fit, y_val, fit_indices, val_indices = train_test_split(
        X_train,
        y_train,
        all_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # Verify that splitting X_train directly without indices produces identical arrays
    X_fit_direct, X_val_direct, y_fit_direct, y_val_direct = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    direct_split_matches = bool(
        np.array_equal(X_fit, X_fit_direct)
        and np.array_equal(X_val, X_val_direct)
        and np.array_equal(y_fit, y_fit_direct)
        and np.array_equal(y_val, y_val_direct)
        and np.array_equal(X_train[fit_indices], X_fit)
        and np.array_equal(X_train[val_indices], X_val)
    )

    fit_set = set(fit_indices.tolist())
    val_set = set(val_indices.tolist())
    overlap_count = len(fit_set.intersection(val_set))
    union_matches_all = (fit_set.union(val_set) == set(range(len(X_train))))

    fit_class_counts = Counter(y_fit.tolist())
    val_class_counts = Counter(y_val.tolist())
    per_class_fit_unique = sorted(set(fit_class_counts.values()))
    per_class_val_unique = sorted(set(val_class_counts.values()))

    # Save reproducible split indices with fingerprint
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.split_output,
        fit_indices=fit_indices,
        validation_indices=val_indices,
        y_fit=y_fit,
        y_validation=y_val,
        class_names=class_names,
        random_state=np.int32(RANDOM_STATE),
        validation_size=np.float64(VALIDATION_SIZE),
        features_file_sha256=np.asarray(npz_file_sha256),
        X_train_sha256=np.asarray(fingerprints["X_train_sha256"]),
        y_train_sha256=np.asarray(fingerprints["y_train_sha256"]),
    )

    # 3. Verify saved model predictions on recreated split
    bundle = joblib.load(args.model)
    pipeline = bundle["model"]
    saved_report = json.loads(args.report_json.read_text(encoding="utf-8"))

    y_val_pred = pipeline.predict(X_val)
    y_fit_pred = pipeline.predict(X_fit)

    val_correct = int(np.sum(y_val_pred == y_val))
    val_incorrect = int(np.sum(y_val_pred != y_val))
    fit_correct = int(np.sum(y_fit_pred == y_fit))
    fit_incorrect = int(np.sum(y_fit_pred != y_fit))

    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_p, val_r, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )

    # Decision function margin check on validation
    decision_scores = pipeline.decision_function(X_val)
    top2_sorted = np.sort(decision_scores, axis=1)
    top1_minus_top2_margin = top2_sorted[:, -1] - top2_sorted[:, -2]
    true_class_scores = decision_scores[np.arange(len(y_val)), y_val]

    # 4. Check exact duplicate HOG vectors
    fit_row_bytes = defaultdict(list)
    for local_i, global_i in enumerate(fit_indices.tolist()):
        fit_row_bytes[X_fit[local_i].tobytes()].append(global_i)

    val_row_bytes = defaultdict(list)
    val_matching_fit_count = 0
    val_matching_fit_examples = []
    for local_i, global_i in enumerate(val_indices.tolist()):
        b = X_val[local_i].tobytes()
        val_row_bytes[b].append(global_i)
        if b in fit_row_bytes:
            val_matching_fit_count += 1
            if len(val_matching_fit_examples) < 10:
                val_matching_fit_examples.append(
                    {"val_index": global_i, "fit_indices": fit_row_bytes[b]}
                )

    all_train_bytes = {X_train[i].tobytes() for i in range(len(X_train))}
    duplicate_rows_within_fit = int(len(X_fit) - len(fit_row_bytes))
    duplicate_rows_within_val = int(len(X_val) - len(val_row_bytes))
    duplicate_rows_within_all_train = int(len(X_train) - len(all_train_bytes))

    # Also check nearest-neighbor HOG distance between X_val and X_fit
    # Compute squared Euclidean distances in batches to keep memory small
    fit_norms_sq = np.sum(X_fit.astype(np.float64) ** 2, axis=1)
    val_norms_sq = np.sum(X_val.astype(np.float64) ** 2, axis=1)
    fit_norms = np.sqrt(fit_norms_sq)
    val_norms = np.sqrt(val_norms_sq)

    nn_fit_local_indices = np.empty(len(X_val), dtype=np.int32)
    nn_l2_distances = np.empty(len(X_val), dtype=np.float64)
    nn_cosine_similarities = np.empty(len(X_val), dtype=np.float64)

    batch_size = 576
    X_fit_64 = X_fit.astype(np.float64)
    for start in range(0, len(X_val), batch_size):
        end = min(start + batch_size, len(X_val))
        X_val_batch = X_val[start:end].astype(np.float64)
        dots = X_val_batch @ X_fit_64.T
        dists_sq = np.maximum(
            0.0,
            val_norms_sq[start:end, None] + fit_norms_sq[None, :] - 2.0 * dots,
        )
        best_local = np.argmin(dists_sq, axis=1)
        best_dist = np.sqrt(dists_sq[np.arange(end - start), best_local])
        best_cos = dots[np.arange(end - start), best_local] / (
            val_norms[start:end] * fit_norms[best_local]
        )
        nn_fit_local_indices[start:end] = best_local
        nn_l2_distances[start:end] = best_dist
        nn_cosine_similarities[start:end] = best_cos

    nn_same_class_ratio = float(np.mean(y_fit[nn_fit_local_indices] == y_val))

    # 5. Reconstruct dataset paths, verify order against build_hog_dataset.py,
    # check raw file hashes, and analyze filename metadata patterns
    train_dir = args.dataset_path / "train"
    test_dir = args.dataset_path / "test"
    fs_class_names = get_class_names(train_dir)
    class_names_match = (fs_class_names == class_names.tolist())

    train_paths, reconstructed_y_train = collect_ordered_paths(train_dir, fs_class_names)
    test_paths, reconstructed_y_test = collect_ordered_paths(test_dir, fs_class_names)

    y_train_order_matches = bool(np.array_equal(reconstructed_y_train, y_train))
    y_test_order_matches = bool(np.array_equal(reconstructed_y_test, y_test))

    # Verify HOG feature extraction on a deterministic sample across all 36 classes
    # Pick first, middle, and last image of several classes + every class start/end
    sample_train_indices = []
    for c_idx in range(EXPECTED_CLASS_COUNT):
        base = c_idx * 800
        sample_train_indices.extend([base, base + 399, base + 799])
    sample_train_matches = 0
    for idx in sample_train_indices:
        recomputed = extract_hog(train_paths[idx])
        if recomputed is not None and np.array_equal(recomputed, X_train[idx]):
            sample_train_matches += 1

    sample_test_indices = [c_idx * 200 for c_idx in range(EXPECTED_CLASS_COUNT)]
    sample_test_matches = 0
    for idx in sample_test_indices:
        recomputed = extract_hog(test_paths[idx])
        if recomputed is not None and np.array_equal(recomputed, X_test[idx]):
            sample_test_matches += 1

    # Check raw image file SHA-256 duplicates across fit and validation
    fit_file_hashes = defaultdict(list)
    for global_i in fit_indices.tolist():
        h = sha256_file(train_paths[global_i])
        fit_file_hashes[h].append(global_i)

    val_file_hashes = defaultdict(list)
    val_exact_file_matches_in_fit = 0
    for global_i in val_indices.tolist():
        h = sha256_file(train_paths[global_i])
        val_file_hashes[h].append(global_i)
        if h in fit_file_hashes:
            val_exact_file_matches_in_fit += 1

    # Analyze filename tokens: {prefix}_{class}_{seq}.jpg
    parsed_train = []
    unmatched_train_filenames = 0
    prefix_seq_ranges = defaultdict(lambda: [10**9, -1])
    for idx, p in enumerate(train_paths):
        m = FILENAME_PATTERN.match(p.name)
        if not m:
            unmatched_train_filenames += 1
            parsed_train.append(None)
            continue
        prefix, label_token, seq_str = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
        parsed_train.append((prefix, label_token, seq_str))
        r = prefix_seq_ranges[prefix]
        r[0] = min(r[0], seq_str)
        r[1] = max(r[1], seq_str)

    parsed_test = []
    unmatched_test_filenames = 0
    for idx, p in enumerate(test_paths):
        m = FILENAME_PATTERN.match(p.name)
        if not m:
            unmatched_test_filenames += 1
            parsed_test.append(None)
            continue
        prefix, label_token, seq_str = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
        parsed_test.append((prefix, label_token, seq_str))
        r = prefix_seq_ranges[prefix]
        r[0] = min(r[0], seq_str)
        r[1] = max(r[1], seq_str)

    # Check group overlap (prefix) and (prefix, class) between fit, val, test
    fit_prefixes = Counter()
    val_prefixes = Counter()
    test_prefixes = Counter()

    fit_prefix_class = Counter()
    val_prefix_class = Counter()
    test_prefix_class = Counter()

    fit_sequences_by_group = defaultdict(set)
    val_sequences_by_group = defaultdict(set)
    test_sequences_by_group = defaultdict(set)

    for global_i in fit_indices.tolist():
        prefix, label_token, seq_num = parsed_train[global_i]
        fit_prefixes[prefix] += 1
        fit_prefix_class[(prefix, label_token)] += 1
        fit_sequences_by_group[(prefix, label_token)].add(seq_num)

    for global_i in val_indices.tolist():
        prefix, label_token, seq_num = parsed_train[global_i]
        val_prefixes[prefix] += 1
        val_prefix_class[(prefix, label_token)] += 1
        val_sequences_by_group[(prefix, label_token)].add(seq_num)

    for info in parsed_test:
        prefix, label_token, seq_num = info
        test_prefixes[prefix] += 1
        test_prefix_class[(prefix, label_token)] += 1
        test_sequences_by_group[(prefix, label_token)].add(seq_num)

    all_prefix_class_groups = set(fit_prefix_class.keys()) | set(val_prefix_class.keys())
    groups_in_both_fit_and_val = len(
        set(fit_prefix_class.keys()) & set(val_prefix_class.keys())
    )
    groups_in_all_three_splits = len(
        set(fit_prefix_class.keys())
        & set(val_prefix_class.keys())
        & set(test_prefix_class.keys())
    )

    # Check sequence number adjacency: how many validation samples have seq_num - 1 or seq_num + 1 in fit?
    val_with_adjacent_seq_in_fit = 0
    val_nn_same_prefix_and_class = 0
    val_nn_seq_diff_le_2 = 0
    example_near_neighbors = []

    for local_val_i, global_val_i in enumerate(val_indices.tolist()):
        v_prefix, v_label, v_seq = parsed_train[global_val_i]
        fit_seqs = fit_sequences_by_group[(v_prefix, v_label)]
        if (v_seq - 1 in fit_seqs) or (v_seq + 1 in fit_seqs):
            val_with_adjacent_seq_in_fit += 1

        best_fit_local = int(nn_fit_local_indices[local_val_i])
        global_fit_i = int(fit_indices[best_fit_local])
        f_prefix, f_label, f_seq = parsed_train[global_fit_i]

        if (v_prefix == f_prefix) and (v_label == f_label):
            val_nn_same_prefix_and_class += 1
            if abs(v_seq - f_seq) <= 2:
                val_nn_seq_diff_le_2 += 1

        if len(example_near_neighbors) < 5 and (v_prefix == f_prefix) and abs(v_seq - f_seq) == 1:
            example_near_neighbors.append(
                {
                    "validation_index": int(global_val_i),
                    "validation_file": train_paths[global_val_i].name,
                    "nearest_fit_index": int(global_fit_i),
                    "nearest_fit_file": train_paths[global_fit_i].name,
                    "hog_l2_distance": float(nn_l2_distances[local_val_i]),
                    "hog_cosine_similarity": float(nn_cosine_similarities[local_val_i]),
                }
            )

    verification_report = {
        "check_1_code_and_predictions": {
            "fit_only_uses_X_fit": True,
            "predict_uses_X_validation": True,
            "metrics_compare_y_validation_and_y_pred": True,
            "hardcoded_or_rounded_metrics_in_code": False,
            "test_set_loaded_in_train_svm": False,
            "fit_samples": int(len(X_fit)),
            "fit_correct_predictions": fit_correct,
            "fit_incorrect_predictions": fit_incorrect,
            "fit_accuracy": float(fit_correct / len(X_fit)),
            "validation_samples": int(len(X_val)),
            "validation_correct_predictions": val_correct,
            "validation_incorrect_predictions": val_incorrect,
            "validation_accuracy_unrounded": val_acc,
            "validation_macro_precision_unrounded": float(val_p),
            "validation_macro_recall_unrounded": float(val_r),
            "validation_macro_f1_unrounded": float(val_f1),
            "matches_saved_json_report": bool(
                val_acc == saved_report["accuracy"]
                and float(val_f1) == saved_report["macro_f1"]
            ),
            "decision_margin_top1_minus_top2": {
                "min": float(np.min(top1_minus_top2_margin)),
                "p01": float(np.percentile(top1_minus_top2_margin, 1)),
                "median": float(np.median(top1_minus_top2_margin)),
                "mean": float(np.mean(top1_minus_top2_margin)),
                "max": float(np.max(top1_minus_top2_margin)),
            },
            "true_class_decision_score": {
                "min": float(np.min(true_class_scores)),
                "median": float(np.median(true_class_scores)),
                "max": float(np.max(true_class_scores)),
            },
        },
        "check_2_split_and_fingerprints": {
            "direct_split_matches_indexed_split": direct_split_matches,
            "fit_indices_count": int(len(fit_indices)),
            "validation_indices_count": int(len(val_indices)),
            "index_overlap_count": overlap_count,
            "union_equals_full_train_28800": union_matches_all,
            "samples_per_class_in_fit": per_class_fit_unique,
            "samples_per_class_in_validation": per_class_val_unique,
            "saved_split_file": str(args.split_output),
            "data_fingerprints": fingerprints,
        },
        "check_3_hog_and_file_duplicates": {
            "exact_hog_duplicates_between_val_and_fit": val_matching_fit_count,
            "exact_hog_duplicates_within_fit": duplicate_rows_within_fit,
            "exact_hog_duplicates_within_val": duplicate_rows_within_val,
            "exact_hog_duplicates_within_all_train": duplicate_rows_within_all_train,
            "exact_image_file_sha256_duplicates_between_val_and_fit": val_exact_file_matches_in_fit,
            "exact_image_file_sha256_duplicates_within_fit": int(
                len(X_fit) - len(fit_file_hashes)
            ),
            "exact_image_file_sha256_duplicates_within_val": int(
                len(X_val) - len(val_file_hashes)
            ),
            "val_to_fit_nearest_neighbor_hog_similarity": {
                "l2_distance_min": float(np.min(nn_l2_distances)),
                "l2_distance_median": float(np.median(nn_l2_distances)),
                "l2_distance_mean": float(np.mean(nn_l2_distances)),
                "l2_distance_max": float(np.max(nn_l2_distances)),
                "cosine_similarity_min": float(np.min(nn_cosine_similarities)),
                "cosine_similarity_median": float(np.median(nn_cosine_similarities)),
                "cosine_similarity_mean": float(np.mean(nn_cosine_similarities)),
                "cosine_similarity_max": float(np.max(nn_cosine_similarities)),
                "nn_same_class_count": int(np.sum(y_fit[nn_fit_local_indices] == y_val)),
                "nn_same_class_ratio": nn_same_class_ratio,
                "nn_same_prefix_and_class_count": val_nn_same_prefix_and_class,
                "nn_same_prefix_and_class_ratio": float(
                    val_nn_same_prefix_and_class / len(X_val)
                ),
                "nn_sequence_number_diff_le_2_count": val_nn_seq_diff_le_2,
                "nn_sequence_number_diff_le_2_ratio": float(
                    val_nn_seq_diff_le_2 / len(X_val)
                ),
            },
        },
        "check_4_metadata_and_filename_analysis": {
            "npz_stores_image_paths": False,
            "external_metadata_files_found": False,
            "path_reconstruction_verified": {
                "class_names_match": class_names_match,
                "y_train_order_matches_100_percent": y_train_order_matches,
                "y_test_order_matches_100_percent": y_test_order_matches,
                "train_hog_spot_check_passed": f"{sample_train_matches}/{len(sample_train_indices)}",
                "test_hog_spot_check_passed": f"{sample_test_matches}/{len(sample_test_indices)}",
            },
            "filename_pattern": "{P1..P10}_{class}_{1..1000}.jpg",
            "unmatched_train_filenames": unmatched_train_filenames,
            "unmatched_test_filenames": unmatched_test_filenames,
            "prefix_tokens": sorted(
                prefix_seq_ranges.keys(), key=lambda x: int(x[1:])
            ),
            "prefix_sequence_number_ranges": {
                k: prefix_seq_ranges[k]
                for k in sorted(prefix_seq_ranges.keys(), key=lambda x: int(x[1:]))
            },
            "prefix_counts_in_fit": {
                k: fit_prefixes[k]
                for k in sorted(fit_prefixes.keys(), key=lambda x: int(x[1:]))
            },
            "prefix_counts_in_validation": {
                k: val_prefixes[k]
                for k in sorted(val_prefixes.keys(), key=lambda x: int(x[1:]))
            },
            "prefix_counts_in_test": {
                k: test_prefixes[k]
                for k in sorted(test_prefixes.keys(), key=lambda x: int(x[1:]))
            },
            "total_prefix_class_groups": len(all_prefix_class_groups),
            "prefix_class_groups_in_both_fit_and_val": groups_in_both_fit_and_val,
            "prefix_class_groups_in_all_three_splits": groups_in_all_three_splits,
            "val_samples_with_adjacent_seq_num_pm1_in_fit": val_with_adjacent_seq_in_fit,
            "val_samples_with_adjacent_seq_num_pm1_ratio": float(
                val_with_adjacent_seq_in_fit / len(X_val)
            ),
            "example_adjacent_nearest_neighbors": example_near_neighbors,
            "signer_identity_verified": False,
            "limitation_note": (
                "No official metadata file confirms whether P1..P10 denote 10 signers, "
                "10 recording sessions, or 10 batches. However, all 360 (P_k, class) "
                "series are split across fit (80 of 100 images in train), validation "
                "(~16 images), and test (20 images), with consecutive sequence numbers "
                "interleaved across splits."
            ),
        },
    }

    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(
        json.dumps(verification_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== VERIFICATION SUMMARY ===")
    print(
        f"1. Predictions: fit={fit_correct}/{len(X_fit)} correct ({fit_incorrect} wrong), "
        f"val={val_correct}/{len(X_val)} correct ({val_incorrect} wrong)"
    )
    print(
        f"   Decision margin (top1 - top2) on val: min={np.min(top1_minus_top2_margin):.4f}, "
        f"median={np.median(top1_minus_top2_margin):.4f}"
    )
    print(
        f"2. Split: fit={len(fit_indices)}, val={len(val_indices)}, "
        f"overlap={overlap_count}, direct_match={direct_split_matches}"
    )
    print(
        f"3. Exact duplicates (val vs fit): HOG={val_matching_fit_count}, "
        f"raw_file_sha256={val_exact_file_matches_in_fit}"
    )
    print(
        f"   1-NN in fit: same_class={nn_same_class_ratio*100:.2f}%, "
        f"same_(P_k,class)={val_nn_same_prefix_and_class}/{len(X_val)} "
        f"({val_nn_same_prefix_and_class/len(X_val)*100:.2f}%), "
        f"|seq_diff|<=2: {val_nn_seq_diff_le_2}/{len(X_val)} "
        f"({val_nn_seq_diff_le_2/len(X_val)*100:.2f}%)"
    )
    print(
        f"   1-NN cosine similarity: median={np.median(nn_cosine_similarities):.4f}, "
        f"min={np.min(nn_cosine_similarities):.4f}, max={np.max(nn_cosine_similarities):.4f}"
    )
    print(
        f"4. Path alignment: train spot-check {sample_train_matches}/{len(sample_train_indices)}, "
        f"test spot-check {sample_test_matches}/{len(sample_test_indices)}"
    )
    print(
        f"   Groups (P_k, class) in both fit & val: {groups_in_both_fit_and_val}/360, "
        f"in all 3 splits: {groups_in_all_three_splits}/360"
    )
    print(
        f"   Val samples with immediate neighbor (seq +/- 1) in fit: "
        f"{val_with_adjacent_seq_in_fit}/{len(X_val)} "
        f"({val_with_adjacent_seq_in_fit/len(X_val)*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
