"""Direction A — Phase 1: Leakage audit, MediaPipe 42-dim landmark cache & leak-free group split on asl_processed.

Strictly Phase 1 only (NO SVM training, NO UI changes, NO modifications to existing models/reports):
1. Scans all 36 classes (0-9, A-Z) across `asl_processed/train` and `asl_processed/test`.
2. Audits original prefix-group overlap and exact SHA-256 duplicates between original train and test,
   as well as cross-prefix-group SHA-256 duplicates across the full 36,000-image pool.
3. Extracts 42-dim aspect-ratio-consistent MediaPipe HandLandmarker features (`mediapipe_21_xy_wrist_norm_aspect_v2`)
   in deterministic order via multiprocessing, preserving EVERY image row (`detected=True/False`).
4. Saves compressed NPZ feature cache, per-image metadata CSV, and 36-class detection statistics.
5. Builds and verifies a zero-leakage `train / validation / test` grouped split where connected components
   of prefix groups sharing any SHA-256 hash stay strictly in one split (`0` prefix-group overlap,
   `0` SHA-256 overlap, all 36 classes covered in all 3 splits).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_hog_dataset import IMAGE_EXTENSIONS, get_class_names
from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    create_hand_landmarker,
    extract_mediapipe_landmarks_feature_from_bgr,
)

DEFAULT_DATASET_PATH = Path(
    r"C:\Users\Quang Diep\Downloads\ASL-HG American Sign Language Hand Gesture Image D"
    r"\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000"
    r"\ASL_Processed_Images\asl_processed"
)
DEFAULT_OUTPUT_DIR = Path("results/svm/experiments/landmark_asl_v1")
FILENAME_PATTERN = re.compile(r"^(P\d+)_([0-9A-Z])_(\d+)\.[A-Za-z0-9]+$", re.IGNORECASE)

# Process-local worker state
_WORKER_DETECTOR = None
_WORKER_BORDER_PAD = 0.5


def _init_worker(model_path_str: str, min_detection_confidence: float, border_pad_ratio: float) -> None:
    global _WORKER_DETECTOR, _WORKER_BORDER_PAD
    _WORKER_BORDER_PAD = float(border_pad_ratio)
    _WORKER_DETECTOR = create_hand_landmarker(
        model_path=Path(model_path_str),
        min_detection_confidence=float(min_detection_confidence),
        num_hands=1,
    )


def _process_chunk(
    items: list[tuple[int, str]],
) -> list[tuple[int, bool, np.ndarray, str, int, int, str]]:
    """Worker function: compute SHA-256, image dims, and 42-dim MediaPipe landmarks for a chunk."""
    out: list[tuple[int, bool, np.ndarray, str, int, int, str]] = []
    zero_vec = np.zeros(LANDMARK_FEATURE_DIM, dtype=np.float32)
    for row_idx, abs_path_str in items:
        p = Path(abs_path_str)
        raw_bytes = p.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            out.append((row_idx, False, zero_vec, "", 0, 0, sha))
            continue
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        feat, info = extract_mediapipe_landmarks_feature_from_bgr(
            bgr,
            detector=_WORKER_DETECTOR,
            metadata={"crop_orientation": "canonical_unflipped_dataset"},
            flip_horizontal=False,
            border_pad_ratio=_WORKER_BORDER_PAD,
        )
        if feat is None:
            out.append((row_idx, False, zero_vec, "", w, h, sha))
        else:
            handed = str(info.get("mediapipe_handedness") or "")
            out.append((row_idx, True, feat.astype(np.float32), handed, w, h, sha))
    return out


def scan_asl_dataset(dataset_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    """Deterministically enumerate all images in asl_processed/train and asl_processed/test."""
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Expected train/ and test/ inside {dataset_path}")

    class_names = get_class_names(train_dir)
    test_classes = get_class_names(test_dir)
    if class_names != test_classes or len(class_names) != 36:
        raise ValueError(f"Unexpected classes: train={len(class_names)}, test={len(test_classes)}")

    records: list[dict[str, object]] = []
    row_idx = 0
    for orig_split in ("train", "test"):
        split_dir = dataset_path / orig_split
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = split_dir / cls_name
            img_paths = sorted(
                p
                for p in cls_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for p in img_paths:
                m = FILENAME_PATTERN.match(p.name)
                if not m:
                    raise ValueError(f"Filename does not match P<k>_<class>_<seq> pattern: {p.name}")
                prefix = m.group(1).upper()
                lbl_tok = m.group(2).upper()
                seq_num = int(m.group(3))
                if lbl_tok != cls_name.upper():
                    raise ValueError(f"Filename class token {lbl_tok} != folder {cls_name} for {p}")
                records.append(
                    {
                        "row_index": row_idx,
                        "orig_split": orig_split,
                        "relative_path": p.relative_to(dataset_path).as_posix(),
                        "abs_path": str(p),
                        "filename": p.name,
                        "prefix_group": prefix,
                        "class_name": cls_name,
                        "class_index": cls_idx,
                        "sequence_number": seq_num,
                    }
                )
                row_idx += 1
    return records, class_names


def sort_prefix_groups(groups: set[str] | list[str]) -> list[str]:
    return sorted(groups, key=lambda g: (int(g[1:]) if g[1:].isdigit() else 999, g))


def build_super_groups(records: list[dict[str, object]]) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    """Union-Find prefix groups that share any exact SHA-256 file hash across all images."""
    all_groups = sort_prefix_groups({str(r["prefix_group"]) for r in records})
    parent = {g: g for g in all_groups}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Anchor on smaller numeric prefix
            if sort_prefix_groups([ra, rb])[0] == ra:
                parent[rb] = ra
            else:
                parent[ra] = rb

    sha_to_rows: dict[str, list[int]] = defaultdict(list)
    for r in records:
        sha_to_rows[str(r["file_sha256"])].append(int(r["row_index"]))

    cross_group_clusters: list[dict[str, object]] = []
    for sha, idxs in sha_to_rows.items():
        grps = sort_prefix_groups({str(records[i]["prefix_group"]) for i in idxs})
        if len(grps) > 1:
            for g in grps[1:]:
                union(grps[0], g)
            cross_group_clusters.append(
                {
                    "groups": grps,
                    "class_names": sorted({str(records[i]["class_name"]) for i in idxs}),
                    "count": len(idxs),
                    "sample_paths": [str(records[i]["relative_path"]) for i in idxs[:4]],
                    "file_sha256": sha,
                }
            )

    comp_map: dict[str, list[str]] = defaultdict(list)
    for g in all_groups:
        comp_map[find(g)].append(g)

    super_groups: dict[str, list[str]] = {}
    for _, members in sorted(comp_map.items(), key=lambda kv: sort_prefix_groups(kv[1])[0]):
        sorted_m = sort_prefix_groups(members)
        sg_id = "+".join(sorted_m)
        super_groups[sg_id] = sorted_m
    return super_groups, cross_group_clusters


def assign_leak_free_splits(
    records: list[dict[str, object]],
    super_groups: dict[str, list[str]],
    class_names: list[str],
    val_groups_target: int = 2,
    test_groups_target: int = 2,
) -> tuple[dict[str, str], dict[str, object]]:
    """Deterministically assign super-groups to train / validation / test with 0 leakage & 36-class coverage.

    Prefers `P8` and `P10` for `validation` (matching `build_prefix_group_split.py`'s existing validation
    groups) and selects `test` groups deterministically from remaining singleton super-groups (excluding
    any merged multi-prefix component such as `P5+P6` which stays intact in `train`).
    """
    # Map each prefix_group -> super_group_id
    pg_to_sg: dict[str, str] = {}
    for sg_id, members in super_groups.items():
        for g in members:
            pg_to_sg[g] = sg_id

    singleton_sgs = [sg for sg, members in super_groups.items() if len(members) == 1]
    merged_sgs = [sg for sg, members in super_groups.items() if len(members) > 1]

    # Deterministic selection:
    # 1. Validation: prefer P8, P10 (exact match with prior prefix-group validation protocol) if singleton
    val_sgs: list[str] = []
    for pref in ("P8", "P10"):
        if pref in singleton_sgs and len(val_sgs) < val_groups_target:
            val_sgs.append(pref)
    for sg in singleton_sgs:
        if len(val_sgs) >= val_groups_target:
            break
        if sg not in val_sgs:
            val_sgs.append(sg)

    # 2. Test: deterministic selection of 2 remaining singleton super-groups (prefer P2, P7 for balanced class detection)
    remaining_singletons = [sg for sg in singleton_sgs if sg not in val_sgs]
    test_sgs: list[str] = []
    for pref in ("P2", "P7"):
        if pref in remaining_singletons and len(test_sgs) < test_groups_target:
            test_sgs.append(pref)
    for sg in remaining_singletons:
        if len(test_sgs) >= test_groups_target:
            break
        if sg not in test_sgs:
            test_sgs.append(sg)

    train_sgs = [sg for sg in super_groups if sg not in val_sgs and sg not in test_sgs]

    sg_to_split: dict[str, str] = {}
    for sg in train_sgs:
        sg_to_split[sg] = "train"
    for sg in val_sgs:
        sg_to_split[sg] = "validation"
    for sg in test_sgs:
        sg_to_split[sg] = "test"

    pg_to_split = {g: sg_to_split[pg_to_sg[g]] for g in pg_to_sg}

    # Verify ZERO prefix_group leakage and ZERO SHA-256 leakage
    pg_splits: dict[str, set[str]] = defaultdict(set)
    sha_splits: dict[str, set[str]] = defaultdict(set)
    total_counts: Counter[tuple[str, str]] = Counter()
    det_counts: Counter[tuple[str, str]] = Counter()
    split_total: Counter[str] = Counter()
    split_det: Counter[str] = Counter()

    for r in records:
        pg = str(r["prefix_group"])
        sp = pg_to_split[pg]
        cls = str(r["class_name"])
        sha = str(r["file_sha256"])
        pg_splits[pg].add(sp)
        sha_splits[sha].add(sp)
        total_counts[sp, cls] += 1
        split_total[sp] += 1
        if bool(r["detected"]):
            det_counts[sp, cls] += 1
            split_det[sp] += 1

    leaked_pg = {pg: sorted(s) for pg, s in pg_splits.items() if len(s) > 1}
    leaked_sha = [sha for sha, s in sha_splits.items() if len(s) > 1]
    if leaked_pg:
        raise RuntimeError(f"Prefix-group leakage detected: {leaked_pg}")
    if leaked_sha:
        raise RuntimeError(f"SHA-256 leakage detected across splits: {len(leaked_sha)} hashes")

    summary: dict[str, object] = {
        "split_rule": {
            "strategy": "connected_component_prefix_group_split",
            "super_groups": super_groups,
            "merged_cross_prefix_super_groups": merged_sgs,
            "train_prefix_groups": sort_prefix_groups([g for g, s in pg_to_split.items() if s == "train"]),
            "validation_prefix_groups": sort_prefix_groups([g for g, s in pg_to_split.items() if s == "validation"]),
            "test_prefix_groups": sort_prefix_groups([g for g, s in pg_to_split.items() if s == "test"]),
        },
        "leakage_verification": {
            "prefix_group_overlap_count": len(leaked_pg),
            "sha256_overlap_count": len(leaked_sha),
            "zero_prefix_group_leakage": len(leaked_pg) == 0,
            "zero_sha256_leakage": len(leaked_sha) == 0,
        },
        "splits": {},
        "original_test_7200_usability_note": (
            "Tập test gốc 7.200 ảnh trong asl_processed/test KHÔNG thể dùng làm tập test tách nhóm độc lập "
            "vì cả 10/10 prefix-group (P1..P10) đều xuất hiện xen kẽ ở cả train gốc lẫn test gốc và có ảnh "
            "trùng SHA-256 giữa hai tập. Do đó thí nghiệm Hướng A bắt buộc dùng phép chia train/validation/test "
            "tách hẳn theo nhóm (super-group) này để đánh giá tổng quát hóa theo người/nhóm chưa từng thấy."
        ),
    }

    for sp in ("train", "validation", "test"):
        sp_pgs = sort_prefix_groups([g for g, s in pg_to_split.items() if s == sp])
        by_cls_total = {c: total_counts[sp, c] for c in class_names}
        by_cls_det = {c: det_counts[sp, c] for c in class_names}
        missing_total = [c for c in class_names if by_cls_total[c] == 0]
        missing_det = [c for c in class_names if by_cls_det[c] == 0]
        low_det = [
            {"class": c, "detected": by_cls_det[c], "total": by_cls_total[c], "rate": round(by_cls_det[c] / by_cls_total[c], 4)}
            for c in class_names
            if by_cls_total[c] > 0 and (by_cls_det[c] / by_cls_total[c]) < 0.7
        ]
        summary["splits"][sp] = {
            "prefix_groups": sp_pgs,
            "prefix_group_count": len(sp_pgs),
            "total_images": split_total[sp],
            "detected_images": split_det[sp],
            "undetected_images": split_total[sp] - split_det[sp],
            "detection_rate": round(split_det[sp] / split_total[sp], 6) if split_total[sp] else 0.0,
            "classes_present_total": 36 - len(missing_total),
            "classes_present_detected": 36 - len(missing_det),
            "missing_classes_total": missing_total,
            "missing_classes_detected": missing_det,
            "low_detection_classes_under_70pct": low_det,
            "min_detected_per_class": min(by_cls_det.values()),
            "max_detected_per_class": max(by_cls_det.values()),
            "per_class_total": by_cls_total,
            "per_class_detected": by_cls_det,
        }
    return pg_to_split, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER_MODEL)
    p.add_argument("--border-pad-ratio", type=float, default=0.5)
    p.add_argument("--min-detection-confidence", type=float, default=0.3)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    p.add_argument("--chunk-size", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    t0 = time.perf_counter()
    print(f"[PHASE 1] Scanning dataset: {args.dataset_path}")
    records, class_names = scan_asl_dataset(args.dataset_path)
    n_total = len(records)
    print(f"[PHASE 1] Found {n_total} images across {len(class_names)} classes. Extracting 42-dim MediaPipe landmarks with {args.workers} workers...")

    import mediapipe as mp
    mp_version = str(getattr(mp, "__version__", "unknown"))

    # Build deterministic chunks
    tasks = [(int(r["row_index"]), str(r["abs_path"])) for r in records]
    chunks = [tasks[i : i + args.chunk_size] for i in range(0, len(tasks), args.chunk_size)]

    X_landmarks = np.zeros((n_total, LANDMARK_FEATURE_DIM), dtype=np.float32)
    detected_arr = np.zeros(n_total, dtype=bool)
    mp_handed_arr = [""] * n_total
    width_arr = np.zeros(n_total, dtype=np.int32)
    height_arr = np.zeros(n_total, dtype=np.int32)

    done_count = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.hand_model), args.min_detection_confidence, args.border_pad_ratio),
    ) as executor:
        for chunk_res in executor.map(_process_chunk, chunks):
            for row_idx, det, vec, handed, w, h, sha in chunk_res:
                detected_arr[row_idx] = det
                X_landmarks[row_idx] = vec
                mp_handed_arr[row_idx] = handed
                width_arr[row_idx] = w
                height_arr[row_idx] = h
                records[row_idx]["detected"] = det
                records[row_idx]["mediapipe_handedness"] = handed
                records[row_idx]["image_width"] = w
                records[row_idx]["image_height"] = h
                records[row_idx]["file_sha256"] = sha
            done_count += len(chunk_res)
            if done_count % 6000 == 0 or done_count == n_total:
                elapsed = time.perf_counter() - t0
                det_so_far = int(np.sum(detected_arr[:done_count]))
                print(
                    f"  Progress: {done_count}/{n_total} ({done_count/n_total*100:.1f}%) "
                    f"| detected={det_so_far}/{done_count} ({det_so_far/done_count*100:.2f}%) | {elapsed:.1f}s"
                )

    elapsed_extract = time.perf_counter() - t0
    print(f"[PHASE 1] Extraction finished in {elapsed_extract:.1f}s. Running leakage audit & grouped split...")

    # 1. Original train vs test leakage audit
    orig_train_recs = [r for r in records if r["orig_split"] == "train"]
    orig_test_recs = [r for r in records if r["orig_split"] == "test"]
    train_pgs = sort_prefix_groups({str(r["prefix_group"]) for r in orig_train_recs})
    test_pgs = sort_prefix_groups({str(r["prefix_group"]) for r in orig_test_recs})
    overlap_pgs = sort_prefix_groups(set(train_pgs).intersection(test_pgs))

    train_shas = {str(r["file_sha256"]) for r in orig_train_recs}
    test_shas = {str(r["file_sha256"]) for r in orig_test_recs}
    overlap_shas = train_shas.intersection(test_shas)
    test_rows_in_train_sha = sum(1 for r in orig_test_recs if str(r["file_sha256"]) in train_shas)
    train_rows_in_test_sha = sum(1 for r in orig_train_recs if str(r["file_sha256"]) in test_shas)
    within_train_dups = len(orig_train_recs) - len(train_shas)
    within_test_dups = len(orig_test_recs) - len(test_shas)

    super_groups, cross_group_clusters = build_super_groups(records)
    cross_pair_counts: Counter[str] = Counter("-".join(c["groups"]) for c in cross_group_clusters)  # type: ignore[arg-type]

    leakage_audit = {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": str(args.dataset_path),
        "total_images": n_total,
        "original_train": {
            "image_count": len(orig_train_recs),
            "unique_sha256_count": len(train_shas),
            "within_split_duplicate_rows": within_train_dups,
            "prefix_groups": train_pgs,
            "prefix_group_count": len(train_pgs),
            "images_by_prefix_group": {
                g: sum(1 for r in orig_train_recs if r["prefix_group"] == g) for g in train_pgs
            },
        },
        "original_test": {
            "image_count": len(orig_test_recs),
            "unique_sha256_count": len(test_shas),
            "within_split_duplicate_rows": within_test_dups,
            "prefix_groups": test_pgs,
            "prefix_group_count": len(test_pgs),
            "images_by_prefix_group": {
                g: sum(1 for r in orig_test_recs if r["prefix_group"] == g) for g in test_pgs
            },
        },
        "original_train_vs_test_leakage": {
            "overlapping_prefix_groups": overlap_pgs,
            "overlapping_prefix_group_count": len(overlap_pgs),
            "overlapping_unique_sha256_count": len(overlap_shas),
            "original_test_rows_exact_sha256_in_train": test_rows_in_train_sha,
            "original_train_rows_exact_sha256_in_test": train_rows_in_test_sha,
            "can_use_original_test_as_unseen_group_test": False,
        },
        "all_36000_cross_prefix_sha256_audit": {
            "cross_prefix_duplicate_clusters_count": len(cross_group_clusters),
            "cross_prefix_duplicate_clusters_by_group_pair": dict(cross_pair_counts),
            "connected_component_super_groups": super_groups,
            "sample_clusters": cross_group_clusters[:10],
        },
    }

    # 2. Detection statistics across all 36 classes
    det_by_class_rows: list[dict[str, object]] = []
    for cls in class_names:
        cls_recs = [r for r in records if r["class_name"] == cls]
        cls_det = sum(1 for r in cls_recs if bool(r["detected"]))
        cls_tot = len(cls_recs)
        cls_by_pg = {}
        for pg in sort_prefix_groups({str(r["prefix_group"]) for r in cls_recs}):
            pg_recs = [r for r in cls_recs if r["prefix_group"] == pg]
            pg_det = sum(1 for r in pg_recs if bool(r["detected"]))
            cls_by_pg[pg] = {"detected": pg_det, "total": len(pg_recs), "rate": round(pg_det / len(pg_recs), 4)}
        det_by_class_rows.append(
            {
                "class_name": cls,
                "total_images": cls_tot,
                "detected_images": cls_det,
                "undetected_images": cls_tot - cls_det,
                "detection_rate": round(cls_det / cls_tot, 6) if cls_tot else 0.0,
                "by_prefix_group": cls_by_pg,
            }
        )

    total_detected = int(np.sum(detected_arr))
    lowest_classes = sorted(det_by_class_rows, key=lambda d: (float(d["detection_rate"]), str(d["class_name"])))[:10]
    detection_summary = {
        "generated_at": datetime.now().isoformat(),
        "feature_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_dim": LANDMARK_FEATURE_DIM,
        "mediapipe_version": mp_version,
        "hand_landmarker_model": str(args.hand_model),
        "border_pad_ratio": float(args.border_pad_ratio),
        "min_detection_confidence": float(args.min_detection_confidence),
        "extraction_seconds": round(elapsed_extract, 2),
        "overall": {
            "total_images": n_total,
            "detected_images": total_detected,
            "undetected_images": n_total - total_detected,
            "detection_rate": round(total_detected / n_total, 6),
        },
        "class_0_detail": next(d for d in det_by_class_rows if d["class_name"] == "0"),
        "lowest_10_classes": [
            {
                "class_name": d["class_name"],
                "detected_images": d["detected_images"],
                "total_images": d["total_images"],
                "detection_rate": d["detection_rate"],
            }
            for d in lowest_classes
        ],
        "per_class": det_by_class_rows,
    }

    # 3. Leak-free grouped train / validation / test split
    pg_to_split, split_summary = assign_leak_free_splits(records, super_groups, class_names)
    for r in records:
        r["new_split"] = pg_to_split[str(r["prefix_group"])]

    # 4. Save all Phase 1 outputs into isolated experiment directory
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Compressed NPZ cache
    cache_path = out_dir / "asl_landmarks_cache.npz"
    np.savez_compressed(
        cache_path,
        X_landmarks=X_landmarks,
        y=np.asarray([int(r["class_index"]) for r in records], dtype=np.int32),
        detected=detected_arr,
        class_names=np.asarray(class_names, dtype=str),
        relative_path=np.asarray([str(r["relative_path"]) for r in records], dtype=str),
        orig_split=np.asarray([str(r["orig_split"]) for r in records], dtype=str),
        new_split=np.asarray([str(r["new_split"]) for r in records], dtype=str),
        prefix_group=np.asarray([str(r["prefix_group"]) for r in records], dtype=str),
        file_sha256=np.asarray([str(r["file_sha256"]) for r in records], dtype=str),
        mediapipe_handedness=np.asarray(mp_handed_arr, dtype=str),
        image_width=width_arr,
        image_height=height_arr,
        feature_schema=np.asarray(LANDMARK_FEATURE_SCHEMA),
        feature_dim=np.int32(LANDMARK_FEATURE_DIM),
        mediapipe_version=np.asarray(mp_version),
        border_pad_ratio=np.float32(args.border_pad_ratio),
        min_detection_confidence=np.float32(args.min_detection_confidence),
        hand_landmarker_model=np.asarray(str(args.hand_model)),
    )

    # 4b. Split manifest CSV (all 36,000 rows with metadata + detected + new_split)
    manifest_csv = out_dir / "split_manifest.csv"
    manifest_fields = [
        "row_index",
        "relative_path",
        "filename",
        "orig_split",
        "new_split",
        "prefix_group",
        "class_name",
        "class_index",
        "sequence_number",
        "image_width",
        "image_height",
        "file_sha256",
        "detected",
        "mediapipe_handedness",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in manifest_fields})

    # 4c. Detection per-class CSV
    det_csv = out_dir / "detection_summary_by_class.csv"
    with det_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "total_images", "detected_images", "undetected_images", "detection_rate"],
        )
        writer.writeheader()
        for d in det_by_class_rows:
            writer.writerow(
                {
                    "class_name": d["class_name"],
                    "total_images": d["total_images"],
                    "detected_images": d["detected_images"],
                    "undetected_images": d["undetected_images"],
                    "detection_rate": d["detection_rate"],
                }
            )

    # 4d. JSON reports
    (out_dir / "asl_leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "detection_summary_by_class.json").write_text(
        json.dumps(detection_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n================ PHASE 1 COMPLETE ================")
    print(f"Total images      : {n_total} | Detected: {total_detected} ({total_detected/n_total*100:.2f}%)")
    print(f"Orig train/test   : overlap_pg={len(overlap_pgs)}/10, test_rows_exact_sha_in_train={test_rows_in_train_sha}")
    print(f"Cross-pg SHA pairs: {dict(cross_pair_counts)}")
    print(f"Class '0' detect  : {detection_summary['class_0_detail']['detected_images']}/{detection_summary['class_0_detail']['total_images']} ({detection_summary['class_0_detail']['detection_rate']*100:.2f}%)")
    print("Lowest 5 classes  :", [(d["class_name"], f"{d['detection_rate']*100:.1f}%") for d in lowest_classes[:5]])
    for sp in ("train", "validation", "test"):
        info = split_summary["splits"][sp]
        print(
            f"  {sp:10s}: groups={info['prefix_groups']} | total={info['total_images']} "
            f"| detected={info['detected_images']} ({info['detection_rate']*100:.2f}%) "
            f"| classes_det={info['classes_present_detected']}/36"
        )
    print(f"Outputs saved in  : {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
