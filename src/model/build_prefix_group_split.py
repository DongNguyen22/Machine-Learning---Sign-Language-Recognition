"""Build and verify a deterministic prefix-group validation split on the 28,800 train samples.

Steps performed:
1. Reconstruct ordered image paths for the 28,800 training images in asl_processed/train.
2. Verify path-to-row alignment against data/processed/svm_hog_features.npz:
   - 100% label match across all 28,800 rows.
   - Exact bitwise HOG match on the first and last image of all 360 (prefix_group, class) blocks (720 images).
3. Save full training image manifest with SHA-256 file hashes to results/svm/svm_train_image_manifest.csv.
4. Create a deterministic prefix-group split using GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42),
   keeping all images with the same filename prefix P_k on one side of the split.
5. Verify class coverage (all 36 classes in both fit and validation), prefix disjointness,
   index disjointness, and SHA-256 / HOG duplicate counts between fit and validation.
6. Save split indices, fingerprints, and metadata to results/svm/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.train_svm import (
    EXPECTED_CLASS_COUNT,
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
        description="Verify image-to-HOG manifest and create a prefix-group validation split."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/svm_hog_features.npz"),
        help="Path to the compressed HOG feature file",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to asl_processed directory",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("results/svm/svm_train_image_manifest.csv"),
        help="Output CSV manifest mapping each row index to file path, prefix group, and SHA-256",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("results/svm/svm_prefix_group_split_indices.npz"),
        help="Output NPZ containing fit_indices and validation_indices for the prefix-group split",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("results/svm/svm_prefix_group_split_metadata.json"),
        help="Output JSON summarizing the prefix-group split verification",
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


def main() -> None:
    args = parse_args()

    print(f"1. Loading HOG training features from: {args.features}")
    X_train, y_train, class_names = load_training_data(args.features)
    features_file_sha256 = sha256_file(args.features)
    X_train_sha256 = sha256_array(X_train)
    y_train_sha256 = sha256_array(y_train)

    train_dir = args.dataset_path / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {train_dir}")

    fs_class_names = get_class_names(train_dir)
    if fs_class_names != class_names.tolist():
        raise ValueError("Filesystem class folder names do not match class_names in NPZ.")

    print("2. Reconstructing ordered image paths and computing SHA-256 hashes...")
    manifest_rows: list[dict[str, object]] = []
    block_boundaries: dict[tuple[str, int], list[int]] = {}

    row_idx = 0
    for class_index, class_name in enumerate(fs_class_names):
        class_dir = train_dir / class_name
        image_paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for p in image_paths:
            match = FILENAME_PATTERN.match(p.name)
            if not match:
                raise ValueError(f"Filename does not match expected pattern: {p.name}")
            prefix_group = match.group(1).upper()
            label_token = match.group(2).upper()
            seq_num = int(match.group(3))
            if label_token != class_name.upper():
                raise ValueError(
                    f"Filename class token {label_token} != folder {class_name} for {p}"
                )

            file_hash = sha256_file(p)
            rel_path = p.relative_to(args.dataset_path).as_posix()
            manifest_rows.append(
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

    if len(manifest_rows) != len(X_train):
        raise ValueError(
            f"Manifest row count {len(manifest_rows)} != X_train count {len(X_train)}"
        )

    reconstructed_labels = np.array(
        [int(r["class_index"]) for r in manifest_rows], dtype=np.int32
    )
    if not np.array_equal(reconstructed_labels, y_train):
        raise ValueError("Reconstructed label order does not match y_train!")

    print(
        f"3. Verifying HOG extraction alignment across all {len(block_boundaries)} "
        "(prefix_group, class) blocks (first and last image of every block)..."
    )
    indices_to_verify: list[int] = []
    for first_idx, last_idx in block_boundaries.values():
        indices_to_verify.append(first_idx)
        if last_idx != first_idx:
            indices_to_verify.append(last_idx)

    verified_count = 0
    for idx in indices_to_verify:
        img_path = Path(manifest_rows[idx]["abs_path"])
        hog_vec = extract_hog(img_path)
        if hog_vec is None or not np.array_equal(hog_vec, X_train[idx]):
            raise ValueError(
                f"HOG feature mismatch at row_index={idx}, file={img_path}"
            )
        verified_count += 1

    print(
        f"   Verified {verified_count}/{len(indices_to_verify)} boundary images "
        f"across all {len(block_boundaries)} (prefix_group, class) blocks: 100% bitwise match."
    )

    # Write manifest CSV
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
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
    with args.manifest_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for r in manifest_rows:
            writer.writerow({k: r[k] for k in csv_fields})

    manifest_sha256 = sha256_file(args.manifest_output)

    # 4. Create deterministic prefix-group split
    prefix_groups = np.array([str(r["prefix_group"]) for r in manifest_rows])
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )
    fit_idx_raw, val_idx_raw = next(gss.split(X_train, y_train, groups=prefix_groups))
    fit_indices = np.sort(fit_idx_raw.astype(np.int32))
    val_indices = np.sort(val_idx_raw.astype(np.int32))

    fit_group_set = set(prefix_groups[fit_indices].tolist())
    val_group_set = set(prefix_groups[val_indices].tolist())
    group_overlap = fit_group_set.intersection(val_group_set)
    if group_overlap:
        raise ValueError(f"Prefix groups overlap between fit and val: {group_overlap}")

    index_overlap = set(fit_indices.tolist()).intersection(set(val_indices.tolist()))
    if index_overlap:
        raise ValueError(f"Row indices overlap between fit and val: {len(index_overlap)}")

    y_fit = y_train[fit_indices]
    y_val = y_train[val_indices]
    fit_classes = np.unique(y_fit)
    val_classes = np.unique(y_val)
    if len(fit_classes) != EXPECTED_CLASS_COUNT or len(val_classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} classes in both splits, "
            f"got fit={len(fit_classes)}, val={len(val_classes)}"
        )

    # Count samples per class and per prefix group
    fit_class_counts = {
        class_names[c]: int(np.sum(y_fit == c)) for c in range(EXPECTED_CLASS_COUNT)
    }
    val_class_counts = {
        class_names[c]: int(np.sum(y_val == c)) for c in range(EXPECTED_CLASS_COUNT)
    }
    fit_prefix_counts = {
        g: int(np.sum(prefix_groups[fit_indices] == g))
        for g in sorted(fit_group_set, key=lambda x: int(x[1:]))
    }
    val_prefix_counts = {
        g: int(np.sum(prefix_groups[val_indices] == g))
        for g in sorted(val_group_set, key=lambda x: int(x[1:]))
    }

    # 5. Check SHA-256 and HOG duplicates across all prefix groups and between fit and validation
    sha_to_rows: dict[str, list[int]] = defaultdict(list)
    hog_to_rows: dict[bytes, list[int]] = defaultdict(list)
    for idx, r in enumerate(manifest_rows):
        sha_to_rows[str(r["file_sha256"])].append(idx)
        hog_to_rows[X_train[idx].tobytes()].append(idx)

    cross_group_sha_clusters = []
    cross_group_pair_counts: Counter[tuple[str, ...]] = Counter()
    for file_hash, idxs in sha_to_rows.items():
        grp_tuple = tuple(
            sorted({str(manifest_rows[i]["prefix_group"]) for i in idxs}, key=lambda x: int(x[1:]))
        )
        if len(grp_tuple) > 1:
            cross_group_pair_counts[grp_tuple] += 1
            cross_group_sha_clusters.append(
                {
                    "groups": list(grp_tuple),
                    "class_name": str(manifest_rows[idxs[0]]["class_name"]),
                    "filenames": [str(manifest_rows[i]["filename"]) for i in idxs],
                    "row_indices": idxs,
                    "file_sha256": file_hash,
                }
            )

    fit_sha_set = {str(manifest_rows[i]["file_sha256"]) for i in fit_indices.tolist()}
    val_sha_list = [str(manifest_rows[i]["file_sha256"]) for i in val_indices.tolist()]
    val_sha_matching_fit = sum(1 for h in val_sha_list if h in fit_sha_set)

    fit_hog_set = {X_train[i].tobytes() for i in fit_indices.tolist()}
    val_hog_matching_fit = sum(
        1 for i in val_indices.tolist() if X_train[i].tobytes() in fit_hog_set
    )

    within_fit_sha_dups = int(len(fit_indices) - len(fit_sha_set))
    within_val_sha_dups = int(len(val_indices) - len(set(val_sha_list)))

    # Compute 1-NN HOG similarity between new validation and new fit
    X_fit = X_train[fit_indices]
    X_val = X_train[val_indices]
    fit_norms_sq = np.sum(X_fit.astype(np.float64) ** 2, axis=1)
    val_norms_sq = np.sum(X_val.astype(np.float64) ** 2, axis=1)
    fit_norms = np.sqrt(fit_norms_sq)
    val_norms = np.sqrt(val_norms_sq)

    nn_l2_distances = np.empty(len(X_val), dtype=np.float64)
    nn_cosine_similarities = np.empty(len(X_val), dtype=np.float64)
    nn_fit_local_indices = np.empty(len(X_val), dtype=np.int32)

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

    nn_same_class_count = int(np.sum(y_fit[nn_fit_local_indices] == y_val))

    # Save NPZ split file
    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    fit_groups_sorted = sorted(fit_group_set, key=lambda x: int(x[1:]))
    val_groups_sorted = sorted(val_group_set, key=lambda x: int(x[1:]))

    np.savez_compressed(
        args.split_output,
        fit_indices=fit_indices,
        validation_indices=val_indices,
        y_fit=y_fit,
        y_validation=y_val,
        fit_groups=np.asarray(fit_groups_sorted),
        validation_groups=np.asarray(val_groups_sorted),
        prefix_groups_all=prefix_groups,
        class_names=class_names,
        split_type=np.asarray("prefix_group_shuffle_split"),
        random_state=np.int32(RANDOM_STATE),
        validation_size=np.float64(VALIDATION_SIZE),
        features_file_sha256=np.asarray(features_file_sha256),
        X_train_sha256=np.asarray(X_train_sha256),
        y_train_sha256=np.asarray(y_train_sha256),
        manifest_sha256=np.asarray(manifest_sha256),
    )

    metadata = {
        "split_name": "prefix_group_validation_split",
        "terminology_note": (
            "P1..P10 are referred to as filename prefix groups ('nhóm tiền tố tên file') "
            "because no local README/metadata file accompanies the dataset archive, and "
            "131 byte-identical files exist across P5 and P6 in classes I and V."
        ),
        "split_rule": {
            "splitter": "sklearn.model_selection.GroupShuffleSplit",
            "n_splits": 1,
            "test_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "group_column": "prefix_group (P1..P10)",
            "selected_before_model_evaluation": True,
        },
        "manifest_verification": {
            "manifest_csv": str(args.manifest_output),
            "manifest_sha256": manifest_sha256,
            "total_train_rows": len(manifest_rows),
            "y_train_exact_match": True,
            "total_prefix_class_blocks": len(block_boundaries),
            "boundary_hog_vectors_verified": verified_count,
            "boundary_hog_bitwise_match_ratio": 1.0,
        },
        "data_fingerprints": {
            "features_file": str(args.features),
            "features_file_sha256": features_file_sha256,
            "X_train_sha256": X_train_sha256,
            "y_train_sha256": y_train_sha256,
        },
        "split_summary": {
            "fit_prefix_groups": fit_groups_sorted,
            "validation_prefix_groups": val_groups_sorted,
            "fit_group_count": len(fit_groups_sorted),
            "validation_group_count": len(val_groups_sorted),
            "prefix_group_overlap_count": len(group_overlap),
            "fit_samples": int(len(fit_indices)),
            "validation_samples": int(len(val_indices)),
            "fit_sample_ratio": float(len(fit_indices) / len(X_train)),
            "validation_sample_ratio": float(len(val_indices) / len(X_train)),
            "index_overlap_count": len(index_overlap),
            "fit_class_count": int(len(fit_classes)),
            "validation_class_count": int(len(val_classes)),
            "fit_samples_per_class_min": int(min(fit_class_counts.values())),
            "fit_samples_per_class_max": int(max(fit_class_counts.values())),
            "validation_samples_per_class_min": int(min(val_class_counts.values())),
            "validation_samples_per_class_max": int(max(val_class_counts.values())),
            "fit_samples_by_prefix": fit_prefix_counts,
            "validation_samples_by_prefix": val_prefix_counts,
            "fit_samples_by_class": fit_class_counts,
            "validation_samples_by_class": val_class_counts,
        },
        "duplicate_and_similarity_audit": {
            "exact_file_sha256_duplicates_between_fit_and_validation": val_sha_matching_fit,
            "exact_hog_duplicates_between_fit_and_validation": val_hog_matching_fit,
            "within_fit_file_sha256_duplicates": within_fit_sha_dups,
            "within_validation_file_sha256_duplicates": within_val_sha_dups,
            "all_train_cross_prefix_sha256_duplicate_clusters_count": len(
                cross_group_sha_clusters
            ),
            "all_train_cross_prefix_duplicate_pairs_by_group": {
                "-".join(k): v for k, v in cross_group_pair_counts.items()
            },
            "cross_prefix_duplicate_note": (
                "Across all 45 pairs of prefix groups in X_train, only (P5, P6) shares "
                "cross-group duplicate files (131 pairs: 67 in class I, 64 in class V). "
                "Because both P5 and P6 are placed in fit under the fixed GroupShuffleSplit "
                "rule, there are 0 file SHA-256 duplicates and 0 HOG duplicates between "
                "fit and validation."
            ),
            "sample_p5_p6_cross_prefix_duplicates_inside_fit": cross_group_sha_clusters[:5],
            "val_to_fit_nearest_neighbor_hog_similarity": {
                "l2_distance_min": float(np.min(nn_l2_distances)),
                "l2_distance_median": float(np.median(nn_l2_distances)),
                "l2_distance_mean": float(np.mean(nn_l2_distances)),
                "l2_distance_max": float(np.max(nn_l2_distances)),
                "cosine_similarity_min": float(np.min(nn_cosine_similarities)),
                "cosine_similarity_median": float(np.median(nn_cosine_similarities)),
                "cosine_similarity_mean": float(np.mean(nn_cosine_similarities)),
                "cosine_similarity_max": float(np.max(nn_cosine_similarities)),
                "nn_same_class_count": nn_same_class_count,
                "nn_same_class_ratio": float(nn_same_class_count / len(X_val)),
            },
        },
        "test_set_limitation": (
            "The original 7,200-sample test set in asl_processed/test contains all 10 "
            "prefix groups (P1..P10) interleaved with the training set. Therefore, while "
            "this prefix-group validation split removes prefix overlap between fit and "
            "validation for hyperparameter selection, the shared team test set remains "
            "an in-distribution / shared-prefix test set rather than an unseen-group test set."
        ),
    }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== PREFIX-GROUP SPLIT SUMMARY ===")
    print(f"Fit groups ({len(fit_groups_sorted)})       : {fit_groups_sorted} -> {len(fit_indices)} samples")
    print(f"Validation groups ({len(val_groups_sorted)}): {val_groups_sorted} -> {len(val_indices)} samples")
    print(
        f"Classes in fit: {len(fit_classes)} (min={min(fit_class_counts.values())}, "
        f"max={max(fit_class_counts.values())})"
    )
    print(
        f"Classes in val: {len(val_classes)} (min={min(val_class_counts.values())}, "
        f"max={max(val_class_counts.values())})"
    )
    print(
        f"Exact duplicates between fit & val: SHA-256={val_sha_matching_fit}, "
        f"HOG={val_hog_matching_fit}"
    )
    print(
        f"1-NN HOG similarity (val -> fit): median cosine={np.median(nn_cosine_similarities):.4f}, "
        f"1-NN same class={nn_same_class_count}/{len(X_val)} ({nn_same_class_count/len(X_val)*100:.2f}%)"
    )
    print(f"Saved manifest to : {args.manifest_output.resolve()}")
    print(f"Saved split NPZ to: {args.split_output.resolve()}")
    print(f"Saved metadata to : {args.metadata_output.resolve()}")


if __name__ == "__main__":
    main()
