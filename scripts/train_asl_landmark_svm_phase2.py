"""Hướng A — Đợt 2: Train & compare Landmark SVM models on Phase 1 cache,
select 1 best config strictly on validation (P8, P10), evaluate ONCE on unseen-group
test (P2, P7), and benchmark side-by-side against HOG Method 3B on real webcam sets.

Safety guarantees:
- Reads Phase 1 cache (`results/svm/experiments/landmark_asl_v1/asl_landmarks_cache.npz`)
  read-only; NEVER overwrites Phase 1 files.
- Never touches any of the 7 existing models in `models/svm/`.
- Only uses SVM (`LinearSVC` and `SVC(kernel='rbf')`) inside
  `Pipeline([('scaler', StandardScaler()), ('clf', ...)])` — NEVER RandomForest.
- Never modifies/confirms webcam sample labels or deletes any images.
- Saves winning model into `models/svm/experiments/landmark_asl_v1/` and Phase 2 reports
  into `results/svm/experiments/landmark_asl_v1/`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import cv2
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_new_webcam_samples import OLD_23_SELECTION_FOLDERS
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.model.train_svm import RANDOM_STATE
from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    canonicalize_saved_crop_bgr,
    create_hand_landmarker,
    extract_mediapipe_landmarks_feature_from_bgr,
    flip_landmark_features_horizontal,
)

DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1" / "asl_landmarks_cache.npz"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v1"
PILOT_DIR = PROJECT_ROOT / "data" / "webcam_pilot"
DIAG_DIR = PROJECT_ROOT / "results" / "svm" / "webcam_diagnostics"
COLLECTION_DIR = PROJECT_ROOT / "data" / "webcam_collection"

HARD_WATCH_CLASSES = ["0", "O", "2", "V", "6", "W", "1", "D", "M", "N", "S", "T", "A", "U"]
SPOTLIGHT_TEST_CLASSES = ["0", "P", "Q", "O", "G"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def load_and_verify_cache(cache_path: Path) -> dict[str, object]:
    """Load Phase 1 NPZ cache and strictly verify schema, split counts, and zero leakage."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"Phase 1 cache not found: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as z:
        feature_schema = str(z["feature_schema"].item())
        feature_dim = int(z["feature_dim"].item())
        mediapipe_version = str(z["mediapipe_version"].item())
        border_pad_ratio = float(z["border_pad_ratio"].item())
        min_det_conf = float(z["min_detection_confidence"].item())
        hand_model = str(z["hand_landmarker_model"].item())

        X_all = np.asarray(z["X_landmarks"], dtype=np.float32)
        y_all = np.asarray(z["y"], dtype=np.int32)
        detected = np.asarray(z["detected"], dtype=bool)
        class_names = [str(c) for c in z["class_names"]]
        new_split = np.asarray(z["new_split"], dtype=str)
        prefix_group = np.asarray(z["prefix_group"], dtype=str)
        file_sha256 = np.asarray(z["file_sha256"], dtype=str)
        handedness = np.asarray(z["mediapipe_handedness"], dtype=str)
        relative_path = np.asarray(z["relative_path"], dtype=str)

    if feature_schema != LANDMARK_FEATURE_SCHEMA:
        raise ValueError(f"Schema mismatch: {feature_schema!r} != {LANDMARK_FEATURE_SCHEMA!r}")
    if feature_dim != LANDMARK_FEATURE_DIM or X_all.shape != (36000, LANDMARK_FEATURE_DIM):
        raise ValueError(f"Unexpected X shape: {X_all.shape}")
    if len(class_names) != 36:
        raise ValueError(f"Expected 36 classes, got {len(class_names)}")

    splits_info: dict[str, dict[str, object]] = {}
    split_groups: dict[str, set[str]] = {}
    split_shas: dict[str, set[str]] = {}
    for sp in ("train", "validation", "test"):
        mask = new_split == sp
        det_mask = mask & detected
        groups = sorted(
            set(prefix_group[mask].tolist()),
            key=lambda s: (int(s[1:]) if len(s) > 1 and s[1:].isdigit() else 999, s),
        )
        split_groups[sp] = set(groups)
        split_shas[sp] = set(file_sha256[mask].tolist())
        splits_info[sp] = {
            "total": int(np.sum(mask)),
            "detected": int(np.sum(det_mask)),
            "undetected": int(np.sum(mask & (~detected))),
            "detection_rate": float(np.sum(det_mask) / np.sum(mask)),
            "prefix_groups": groups,
            "mask": mask,
            "det_mask": det_mask,
        }

    # Strict assertions matching Phase 1 verified numbers
    assert splits_info["train"]["prefix_groups"] == ["P1", "P3", "P4", "P5", "P6", "P9"], splits_info["train"]["prefix_groups"]
    assert splits_info["validation"]["prefix_groups"] == ["P8", "P10"], splits_info["validation"]["prefix_groups"]
    assert splits_info["test"]["prefix_groups"] == ["P2", "P7"], splits_info["test"]["prefix_groups"]
    assert splits_info["train"]["detected"] == 20803 and splits_info["train"]["total"] == 21600
    assert splits_info["validation"]["detected"] == 6932 and splits_info["validation"]["total"] == 7200
    assert splits_info["test"]["detected"] == 6742 and splits_info["test"]["total"] == 7200

    # Zero leakage checks
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        g_overlap = split_groups[a] & split_groups[b]
        s_overlap = split_shas[a] & split_shas[b]
        if g_overlap:
            raise RuntimeError(f"Prefix-group leakage between {a} and {b}: {sorted(g_overlap)}")
        if s_overlap:
            raise RuntimeError(f"SHA-256 leakage between {a} and {b}: {len(s_overlap)} hashes")

    if not np.isfinite(X_all[detected]).all():
        raise ValueError("Detected landmark features contain NaN or Inf!")

    return {
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "feature_schema": feature_schema,
        "feature_dim": feature_dim,
        "mediapipe_version": mediapipe_version,
        "border_pad_ratio": border_pad_ratio,
        "min_detection_confidence": min_det_conf,
        "hand_landmarker_model": hand_model,
        "X_all": X_all,
        "y_all": y_all,
        "detected": detected,
        "class_names": class_names,
        "new_split": new_split,
        "prefix_group": prefix_group,
        "file_sha256": file_sha256,
        "handedness": handedness,
        "relative_path": relative_path,
        "splits": splits_info,
    }


def summarize_handedness(cache: dict[str, object]) -> dict[str, object]:
    handedness = cache["handedness"]  # type: ignore[assignment]
    prefix_group = cache["prefix_group"]  # type: ignore[assignment]
    splits = cache["splits"]  # type: ignore[assignment]

    out: dict[str, object] = {
        "note": (
            "mediapipe_handedness được MediaPipe Tasks HandLandmarker gán trên ảnh gốc chưa lật "
            "của asl_processed (chỉ tính trên mẫu detected=True)."
        ),
        "by_split": {},
        "by_prefix_group": {},
    }

    for sp in ("train", "validation", "test"):
        det_mask = splits[sp]["det_mask"]
        vals = [str(v) if str(v) else "Unknown" for v in handedness[det_mask].tolist()]
        cnt = Counter(vals)
        n = len(vals)
        out["by_split"][sp] = {
            "detected_total": n,
            "counts": dict(sorted(cnt.items())),
            "percentages": {k: round(v / n * 100.0, 2) for k, v in sorted(cnt.items())},
            "left_count": int(cnt.get("Left", 0)),
            "right_count": int(cnt.get("Right", 0)),
            "other_count": int(n - cnt.get("Left", 0) - cnt.get("Right", 0)),
            "left_pct": round(cnt.get("Left", 0) / n * 100.0, 2) if n else 0.0,
            "right_pct": round(cnt.get("Right", 0) / n * 100.0, 2) if n else 0.0,
        }

    all_det = cache["detected"]  # type: ignore[assignment]
    for pg in sorted(set(prefix_group.tolist()), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
        m = (prefix_group == pg) & all_det
        vals = [str(v) if str(v) else "Unknown" for v in handedness[m].tolist()]
        cnt = Counter(vals)
        n = len(vals)
        out["by_prefix_group"][pg] = {
            "detected_total": n,
            "left_count": int(cnt.get("Left", 0)),
            "right_count": int(cnt.get("Right", 0)),
            "left_pct": round(cnt.get("Left", 0) / n * 100.0, 2) if n else 0.0,
            "right_pct": round(cnt.get("Right", 0) / n * 100.0, 2) if n else 0.0,
        }

    return out


def build_candidate_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for aug in (False, True):
        for c_val in (0.1, 1.0, 10.0):
            specs.append(
                {
                    "candidate_id": f"linear_C{c_val:g}_aug{'1' if aug else '0'}",
                    "model_family": "LinearSVC",
                    "kernel": "linear",
                    "C": float(c_val),
                    "gamma": None,
                    "augment_horizontal_flip": aug,
                }
            )
        for c_val in (1.0, 10.0, 30.0):
            for gamma in ("scale", 0.05):
                specs.append(
                    {
                        "candidate_id": f"rbf_C{c_val:g}_g{gamma}_aug{'1' if aug else '0'}",
                        "model_family": "SVC(rbf)",
                        "kernel": "rbf",
                        "C": float(c_val),
                        "gamma": gamma,
                        "augment_horizontal_flip": aug,
                    }
                )
    return specs


def build_svm_pipeline(spec: dict[str, object], max_iter: int = 10_000) -> Pipeline:
    if spec["kernel"] == "linear":
        clf = LinearSVC(
            C=float(spec["C"]),
            dual="auto",
            max_iter=max_iter,
            random_state=RANDOM_STATE,
        )
    elif spec["kernel"] == "rbf":
        clf = SVC(
            C=float(spec["C"]),
            kernel="rbf",
            gamma=spec["gamma"],  # type: ignore[arg-type]
            decision_function_shape="ovr",
            probability=False,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported kernel: {spec['kernel']!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def eval_predictions_on_split(
    y_true_det: np.ndarray,
    y_pred_det: np.ndarray,
    total_split_count: int,
    class_names: list[str],
) -> dict[str, object]:
    n_det = int(len(y_true_det))
    n_correct = int(np.sum(y_true_det == y_pred_det))
    cond_acc = float(n_correct / n_det) if n_det else 0.0
    cond_macro_f1 = float(
        f1_score(y_true_det, y_pred_det, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    e2e_acc = float(n_correct / total_split_count) if total_split_count else 0.0

    _, recalls, f1s, _ = precision_recall_fscore_support(
        y_true_det, y_pred_det, labels=list(range(len(class_names))), zero_division=0
    )
    watch_recall = {
        cls: round(float(recalls[class_names.index(cls)]), 4)
        for cls in HARD_WATCH_CLASSES
        if cls in class_names
    }
    return {
        "detected_count": n_det,
        "total_count": int(total_split_count),
        "correct_count": n_correct,
        "cond_accuracy": cond_acc,
        "cond_macro_f1": cond_macro_f1,
        "end_to_end_accuracy": e2e_acc,
        "hard_watch_recall": watch_recall,
        "hard_watch_mean_recall": float(np.mean(list(watch_recall.values()))),
        "per_class_f1": {class_names[i]: float(f1s[i]) for i in range(len(class_names))},
    }


def evaluate_detailed_test(
    pipe: Pipeline,
    X_test_det: np.ndarray,
    y_test_det: np.ndarray,
    y_test_all: np.ndarray,
    test_detected_flags: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    y_pred_det = np.asarray(pipe.predict(X_test_det), dtype=np.int32)
    scores_det = np.asarray(pipe.decision_function(X_test_det), dtype=np.float64)
    infer_s = time.perf_counter() - t0

    n_total = int(len(y_test_all))
    n_det = int(len(y_test_det))
    n_undet = int(n_total - n_det)
    n_correct = int(np.sum(y_test_det == y_pred_det))

    cond_acc = float(accuracy_score(y_test_det, y_pred_det))
    cond_macro_f1 = float(
        f1_score(y_test_det, y_pred_det, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    e2e_acc = float(n_correct / n_total)

    # Secondary bilateral sanity check on mirrored test features (reported separately)
    X_test_mirrored = flip_landmark_features_horizontal(X_test_det)
    y_pred_mir = np.asarray(pipe.predict(X_test_mirrored), dtype=np.int32)
    mir_correct = int(np.sum(y_test_det == y_pred_mir))
    mir_cond_acc = float(mir_correct / n_det)
    mir_cond_macro_f1 = float(
        f1_score(y_test_det, y_pred_mir, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    mir_e2e_acc = float(mir_correct / n_total)

    prec, rec, f1s, sup_det = precision_recall_fscore_support(
        y_test_det, y_pred_det, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_rows: list[dict[str, object]] = []
    for i, cname in enumerate(class_names):
        tot_c = int(np.sum(y_test_all == i))
        det_c = int(sup_det[i])
        undet_c = int(tot_c - det_c)
        corr_c = int(np.sum((y_test_det == i) & (y_pred_det == i)))
        per_class_rows.append(
            {
                "class_index": i,
                "class_name": cname,
                "support_total": tot_c,
                "support_detected": det_c,
                "undetected_count": undet_c,
                "detection_rate": round(det_c / tot_c, 6) if tot_c else 0.0,
                "correct_count": corr_c,
                "cond_precision": round(float(prec[i]), 6),
                "cond_recall": round(float(rec[i]), 6),
                "cond_f1": round(float(f1s[i]), 6),
                "end_to_end_recall": round(corr_c / tot_c, 6) if tot_c else 0.0,
            }
        )

    cm = confusion_matrix(y_test_det, y_pred_det, labels=list(range(len(class_names))))
    confusions: list[dict[str, object]] = []
    for ti in range(len(class_names)):
        row_total = int(cm[ti].sum())
        for pi in range(len(class_names)):
            if ti == pi:
                continue
            cnt = int(cm[ti, pi])
            if cnt > 0:
                confusions.append(
                    {
                        "true_label": class_names[ti],
                        "predicted_label": class_names[pi],
                        "count": cnt,
                        "pct_of_true_detected": round(cnt / row_total * 100.0, 2) if row_total else 0.0,
                    }
                )
    confusions.sort(key=lambda r: (-int(r["count"]), str(r["true_label"]), str(r["predicted_label"])))
    top_10_confusions = confusions[:10]

    spotlight = {
        r["class_name"]: r for r in per_class_rows if r["class_name"] in SPOTLIGHT_TEST_CLASSES
    }
    weakest_by_cond_f1 = sorted(per_class_rows, key=lambda r: (float(r["cond_f1"]), float(r["end_to_end_recall"])))[:10]
    weakest_by_e2e = sorted(per_class_rows, key=lambda r: (float(r["end_to_end_recall"]), float(r["cond_f1"])))[:10]

    # Golden slice (first 64 detected test samples) for exact bit-for-bit reload test
    golden_n = min(64, n_det)
    golden_check = {
        "slice_size": golden_n,
        "y_pred_first_64": y_pred_det[:golden_n].tolist(),
        "decision_scores_first_64_sha256": sha256_array(scores_det[:golden_n]),
        "full_y_pred_sha256": sha256_array(y_pred_det),
        "full_decision_scores_sha256": sha256_array(scores_det),
    }

    summary = {
        "split": "test",
        "prefix_groups": ["P2", "P7"],
        "total_samples": n_total,
        "detected_samples": n_det,
        "undetected_samples": n_undet,
        "detection_rate": round(n_det / n_total, 6),
        "correct_samples": n_correct,
        "cond_accuracy": round(cond_acc, 6),
        "cond_macro_f1": round(cond_macro_f1, 6),
        "end_to_end_accuracy": round(e2e_acc, 6),
        "inference_seconds_all_detected": round(infer_s, 4),
        "inference_ms_per_detected_sample": round(infer_s / n_det * 1000.0, 4) if n_det else 0.0,
        "test_mirrored_bilateral_check": {
            "correct_samples": mir_correct,
            "cond_accuracy": round(mir_cond_acc, 6),
            "cond_macro_f1": round(mir_cond_macro_f1, 6),
            "end_to_end_accuracy": round(mir_e2e_acc, 6),
        },
        "top_10_confusions": top_10_confusions,
        "spotlight_classes": spotlight,
        "weakest_10_by_cond_f1": weakest_by_cond_f1,
        "weakest_10_by_end_to_end_recall": weakest_by_e2e,
        "per_class": per_class_rows,
        "golden_reload_verification": golden_check,
    }
    return summary, per_class_rows, y_pred_det, scores_det


def predict_landmark_top_k(
    pipe: Pipeline,
    feat_42: np.ndarray,
    class_names: list[str],
    top_k: int = 3,
) -> dict[str, object]:
    X = feat_42.reshape(1, -1)
    pred_idx = int(pipe.predict(X)[0])
    scores = np.asarray(pipe.decision_function(X)[0], dtype=np.float64)
    clf = pipe.named_steps["clf"]
    classes = np.asarray(clf.classes_, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    top_list = []
    for rank, pos in enumerate(order[:top_k], start=1):
        cid = int(classes[int(pos)])
        top_list.append(
            {
                "rank": rank,
                "label": str(class_names[cid]),
                "decision_score": round(float(scores[int(pos)]), 4),
            }
        )
    # Note: for SVC(decision_function_shape='ovr'), predict() uses OvO vote while decision_function
    # uses break_ties=False OvR monotonic transform; always report official pipe.predict() label!
    pred_label = str(class_names[pred_idx])
    pos_pred = int(np.where(classes == pred_idx)[0][0])
    pred_score = float(scores[pos_pred])
    margin = float(scores[int(order[0])] - scores[int(order[1])]) if len(order) >= 2 else 0.0
    return {
        "predicted_label": pred_label,
        "predicted_score": round(pred_score, 4),
        "margin_top1_minus_top2": round(margin, 4),
        "top_k": top_list,
    }


def collect_webcam_sample_sets() -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collect real webcam sets strictly separated by source and confirmation status."""
    # 1. data/webcam_pilot (15 confirmed samples, P01/S01, left hand)
    pilot_dirs = sorted(
        [d for d in PILOT_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )
    pilot_items = []
    for d in pilot_dirs:
        meta = json.loads((d / "diagnostic_metadata.json").read_text(encoding="utf-8"))
        assert bool(meta.get("label_confirmed", False)) is True, f"Unconfirmed pilot sample: {d.name}"
        pilot_items.append((d, meta, str(meta["verified_true_label"]).strip().upper()))

    # 2. results/svm/webcam_diagnostics (60 dirs total = 55 confirmed [23 old + 32 new] + 5 unconfirmed)
    diag_dirs = sorted(
        [d for d in DIAG_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )
    diag_confirmed_32 = []
    diag_old_23 = []
    diag_legacy_55 = []
    diag_unconfirmed_excluded = []

    for d in diag_dirs:
        mp = d / "diagnostic_metadata.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if not bool(meta.get("label_confirmed", False)):
            diag_unconfirmed_excluded.append(
                {
                    "folder_name": d.name,
                    "label_confirmed": False,
                    "participant_id": meta.get("participant_id"),
                    "intended_sign": meta.get("intended_sign"),
                }
            )
            continue
        lbl = str(meta.get("verified_true_label") or "").strip().upper()
        if not lbl:
            raise ValueError(f"Confirmed sample missing verified_true_label: {d.name}")
        item = (d, meta, lbl)
        diag_legacy_55.append(item)
        if d.name in OLD_23_SELECTION_FOLDERS:
            diag_old_23.append(item)
        else:
            diag_confirmed_32.append(item)

    # Check data/webcam_collection (1 unconfirmed sample — never mixed in)
    collection_unconfirmed = []
    if COLLECTION_DIR.exists():
        for d in sorted([p for p in COLLECTION_DIR.iterdir() if p.is_dir() and p.name.startswith("sample_")]):
            mp = d / "diagnostic_metadata.json"
            if mp.exists():
                meta = json.loads(mp.read_text(encoding="utf-8"))
                collection_unconfirmed.append(
                    {
                        "folder_name": d.name,
                        "label_confirmed": bool(meta.get("label_confirmed", False)),
                        "participant_id": meta.get("participant_id"),
                        "intended_sign": meta.get("intended_sign"),
                    }
                )

    assert len(pilot_items) == 15, f"Expected 15 pilot samples, got {len(pilot_items)}"
    assert len(diag_confirmed_32) == 32, f"Expected 32 confirmed new samples, got {len(diag_confirmed_32)}"
    assert len(diag_old_23) == 23, f"Expected 23 old selection samples, got {len(diag_old_23)}"
    assert len(diag_legacy_55) == 55, f"Expected 55 legacy confirmed samples, got {len(diag_legacy_55)}"

    datasets = [
        {
            "set_id": "webcam_pilot_15",
            "title": "data/webcam_pilot (15 mẫu A/B/C/D đã xác nhận, P01/S01, tay trái)",
            "source_dir": str(PILOT_DIR.relative_to(PROJECT_ROOT).as_posix()),
            "label_confirmed_status": "15/15 label_confirmed=True (có đủ participant_id=P01, session_id=S01, gesture_instance_id, handedness=left)",
            "items": pilot_items,
        },
        {
            "set_id": "webcam_diagnostics_confirmed_32",
            "title": "results/svm/webcam_diagnostics — 32 mẫu độc lập A/B/C/D đã xác nhận (loại trừ 23 mẫu cũ)",
            "source_dir": str(DIAG_DIR.relative_to(PROJECT_ROOT).as_posix()),
            "label_confirmed_status": "32/32 label_confirmed=True (thiếu metadata participant_id/session_id/gesture_instance_id; chụp sau khi chốt Cách 3B)",
            "items": diag_confirmed_32,
        },
        {
            "set_id": "webcam_diagnostics_old_23",
            "title": "results/svm/webcam_diagnostics — 23 mẫu cũ A/B/C/D đã xác nhận (tập từng dùng chọn Cách 3B)",
            "source_dir": str(DIAG_DIR.relative_to(PROJECT_ROOT).as_posix()),
            "label_confirmed_status": "23/23 label_confirmed=True (thiếu metadata participant_id/session_id; tập từng dùng chọn HOG Cách 3B)",
            "items": diag_old_23,
        },
        {
            "set_id": "webcam_diagnostics_legacy_55",
            "title": "results/svm/webcam_diagnostics — Toàn bộ 55 mẫu cũ A/B/C/D đã xác nhận (23 cũ + 32 độc lập)",
            "source_dir": str(DIAG_DIR.relative_to(PROJECT_ROOT).as_posix()),
            "label_confirmed_status": "55/55 label_confirmed=True (thiếu metadata participant_id/session_id/gesture_instance_id; gồm 23 mẫu từng dùng chọn Cách 3B + 32 mẫu độc lập)",
            "items": diag_legacy_55,
        },
    ]

    excluded_info = {
        "webcam_diagnostics_unconfirmed_excluded": diag_unconfirmed_excluded,
        "webcam_collection_unconfirmed_excluded": collection_unconfirmed,
        "note": (
            "Các mẫu chưa xác nhận nhãn (label_confirmed=False) được loại khỏi thống kê độ chính xác "
            "và tuyệt đối không bị tự ý xác nhận hay sửa nhãn."
        ),
    }
    return datasets, excluded_info


def run_webcam_benchmark(
    landmark_pipe: Pipeline,
    class_names: list[str],
    hand_model_path: Path,
    border_pad_ratio: float,
    min_detection_confidence: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    datasets, excluded_info = collect_webcam_sample_sets()
    hog_predictor = SVMSignPredictor(model_path=PROJECT_ROOT / DEFAULT_MODEL_PATH)
    detector = create_hand_landmarker(
        model_path=hand_model_path,
        min_detection_confidence=min_detection_confidence,
        num_hands=1,
    )

    # Cache per-folder inference so samples shared between 32/23 and 55 are evaluated identically once
    per_folder_cache: dict[str, dict[str, object]] = {}
    all_csv_rows: list[dict[str, object]] = []
    sets_report: dict[str, object] = {}

    try:
        for ds in datasets:
            set_id = str(ds["set_id"])
            items = ds["items"]
            sample_rows: list[dict[str, object]] = []

            for idx, (sdir, meta, true_label) in enumerate(items, start=1):
                key = str(sdir.resolve())
                if key not in per_folder_cache:
                    crop_path = sdir / "received_crop.png"
                    crop_bgr = cv2.imread(str(crop_path))
                    if crop_bgr is None:
                        raise ValueError(f"Cannot read crop image: {crop_path}")

                    # Landmark SVM (exact Phase 1 formula)
                    feat_42, mp_info = extract_mediapipe_landmarks_feature_from_bgr(
                        crop_bgr,
                        detector=detector,
                        metadata=meta,
                        flip_horizontal=False,
                        border_pad_ratio=border_pad_ratio,
                    )
                    if feat_42 is None:
                        lm_detected = False
                        lm_pred = None
                        lm_score = None
                        lm_margin = None
                        lm_top3: list[dict[str, object]] = []
                        mp_hand = None
                    else:
                        lm_detected = True
                        lm_res = predict_landmark_top_k(landmark_pipe, feat_42, class_names, top_k=3)
                        lm_pred = str(lm_res["predicted_label"])
                        lm_score = lm_res["predicted_score"]
                        lm_margin = lm_res["margin_top1_minus_top2"]
                        lm_top3 = lm_res["top_k"]  # type: ignore[assignment]
                        mp_hand = mp_info.get("mediapipe_handedness")

                    # HOG Method 3B (exact same canonical crop -> single flip + soft bg suppression)
                    canonical_bgr = canonicalize_saved_crop_bgr(crop_bgr, metadata=meta)
                    hog_res = hog_predictor.predict_image_bgr(
                        canonical_bgr,
                        top_k=3,
                        include_diagnostics=False,
                        experimental_webcam_mode=True,
                        already_flipped=False,
                    )
                    hog_pred = str(hog_res["predicted_label"])
                    hog_score = round(float(hog_res["predicted_decision_score"]), 4)
                    hog_top3 = [
                        {
                            "rank": int(it["rank"]),
                            "label": str(it["label"]),
                            "decision_score": round(float(it["decision_score"]), 4),
                        }
                        for it in hog_res["top_k_predictions"]  # type: ignore[union-attr]
                    ]

                    per_folder_cache[key] = {
                        "folder_name": sdir.name,
                        "relative_dir": sdir.relative_to(PROJECT_ROOT).as_posix(),
                        "true_label": true_label,
                        "label_confirmed": bool(meta.get("label_confirmed", False)),
                        "participant_id": meta.get("participant_id") or "",
                        "session_id": meta.get("session_id") or "",
                        "gesture_instance_id": meta.get("gesture_instance_id") or "",
                        "handedness_meta": meta.get("handedness") or "",
                        "crop_orientation": meta.get("crop_orientation") or "",
                        "flip_input_for_model": bool(meta.get("flip_input_for_model", False)),
                        "crop_shape": f"{crop_bgr.shape[1]}x{crop_bgr.shape[0]}",
                        "landmark_hand_detected": lm_detected,
                        "mediapipe_handedness": mp_hand or "",
                        "landmark_pred": lm_pred or "__NO_HAND__",
                        "landmark_correct": (lm_detected and lm_pred == true_label),
                        "landmark_score": lm_score,
                        "landmark_margin": lm_margin,
                        "landmark_top3": lm_top3,
                        "hog_3b_pred": hog_pred,
                        "hog_3b_correct": (hog_pred == true_label),
                        "hog_3b_score": hog_score,
                        "hog_3b_top3": hog_top3,
                    }

                rec = dict(per_folder_cache[key])
                rec["stt"] = idx
                rec["set_id"] = set_id
                sample_rows.append(rec)

                # For CSV avoid duplicating 55 rows if we keep set_id column (keep all 4 sets clearly tagged)
                all_csv_rows.append(
                    {
                        "set_id": set_id,
                        "stt": idx,
                        "folder_name": rec["folder_name"],
                        "true_label": rec["true_label"],
                        "participant_id": rec["participant_id"],
                        "gesture_instance_id": rec["gesture_instance_id"],
                        "handedness_meta": rec["handedness_meta"],
                        "mediapipe_handedness": rec["mediapipe_handedness"],
                        "landmark_hand_detected": rec["landmark_hand_detected"],
                        "landmark_pred": rec["landmark_pred"],
                        "landmark_correct": rec["landmark_correct"],
                        "landmark_score": rec["landmark_score"],
                        "landmark_top3": "; ".join(
                            f"{t['label']}:{t['decision_score']}" for t in rec["landmark_top3"]
                        ),
                        "hog_3b_pred": rec["hog_3b_pred"],
                        "hog_3b_correct": rec["hog_3b_correct"],
                        "hog_3b_score": rec["hog_3b_score"],
                    }
                )

            n_total = len(sample_rows)
            det_rows = [r for r in sample_rows if r["landmark_hand_detected"]]
            n_det = len(det_rows)
            lm_corr = sum(1 for r in sample_rows if r["landmark_correct"])
            hog_corr = sum(1 for r in sample_rows if r["hog_3b_correct"])
            hog_corr_on_det = sum(1 for r in det_rows if r["hog_3b_correct"])

            classes_present = sorted({str(r["true_label"]) for r in sample_rows})
            per_class: dict[str, object] = {}
            for cls in classes_present:
                c_rows = [r for r in sample_rows if r["true_label"] == cls]
                c_det = [r for r in c_rows if r["landmark_hand_detected"]]
                c_lm_ok = sum(1 for r in c_rows if r["landmark_correct"])
                c_hog_ok = sum(1 for r in c_rows if r["hog_3b_correct"])
                per_class[cls] = {
                    "total": len(c_rows),
                    "landmark_detected": len(c_det),
                    "landmark_correct": c_lm_ok,
                    "landmark_e2e_pct": round(c_lm_ok / len(c_rows) * 100.0, 2) if c_rows else 0.0,
                    "landmark_cond_pct": round(c_lm_ok / len(c_det) * 100.0, 2) if c_det else 0.0,
                    "landmark_pred_dist": dict(Counter(str(r["landmark_pred"]) for r in c_rows)),
                    "hog_3b_correct": c_hog_ok,
                    "hog_3b_e2e_pct": round(c_hog_ok / len(c_rows) * 100.0, 2) if c_rows else 0.0,
                    "hog_3b_pred_dist": dict(Counter(str(r["hog_3b_pred"]) for r in c_rows)),
                }

            landmark_errors_or_misses = [
                {
                    "stt": r["stt"],
                    "folder_name": r["folder_name"],
                    "gesture_instance_id": r["gesture_instance_id"],
                    "true_label": r["true_label"],
                    "hand_detected": r["landmark_hand_detected"],
                    "mediapipe_handedness": r["mediapipe_handedness"],
                    "landmark_pred": r["landmark_pred"],
                    "landmark_top3": r["landmark_top3"],
                    "hog_3b_pred": r["hog_3b_pred"],
                    "hog_3b_correct": r["hog_3b_correct"],
                }
                for r in sample_rows
                if not r["landmark_correct"]
            ]

            sets_report[set_id] = {
                "title": ds["title"],
                "source_dir": ds["source_dir"],
                "label_confirmed_status": ds["label_confirmed_status"],
                "total_samples": n_total,
                "landmark_detected_samples": n_det,
                "landmark_undetected_samples": n_total - n_det,
                "landmark_detection_rate_pct": round(n_det / n_total * 100.0, 2) if n_total else 0.0,
                "landmark_correct": lm_corr,
                "landmark_end_to_end_accuracy_pct": round(lm_corr / n_total * 100.0, 2) if n_total else 0.0,
                "landmark_conditional_accuracy_pct": round(lm_corr / n_det * 100.0, 2) if n_det else 0.0,
                "hog_3b_correct": hog_corr,
                "hog_3b_end_to_end_accuracy_pct": round(hog_corr / n_total * 100.0, 2) if n_total else 0.0,
                "hog_3b_correct_on_landmark_detected_subset": hog_corr_on_det,
                "hog_3b_accuracy_on_landmark_detected_subset_pct": (
                    round(hog_corr_on_det / n_det * 100.0, 2) if n_det else 0.0
                ),
                "per_class": per_class,
                "landmark_errors_or_misses": landmark_errors_or_misses,
                "samples": sample_rows,
            }
    finally:
        detector.close()

    benchmark_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "landmark_feature_schema": LANDMARK_FEATURE_SCHEMA,
        "border_pad_ratio": float(border_pad_ratio),
        "min_detection_confidence": float(min_detection_confidence),
        "hog_baseline_model": str(DEFAULT_MODEL_PATH.as_posix()),
        "hog_baseline_mode": "method_3b",
        "limitation_note": (
            "Toàn bộ dữ liệu webcam đã xác nhận nhãn hiện có chỉ gồm 4 lớp (A, B, C, D) từ 1 người chụp "
            "(và 55 mẫu cũ thiếu metadata người chụp). Chưa có tập kiểm thử webcam 36 lớp nhiều người, "
            "nên tuyệt đối chưa kết luận độ chính xác thực tế cho đủ 36 lớp trên webcam."
        ),
        "excluded_unconfirmed_sets": excluded_info,
        "sets": sets_report,
    }
    return benchmark_report, all_csv_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--max-iter", type=int, default=10_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    print("================ HƯỚNG A — ĐỢT 2: LANDMARK SVM EXPERIMENT ================", flush=True)
    cache = load_and_verify_cache(args.cache)
    class_names: list[str] = cache["class_names"]  # type: ignore[assignment]
    X_all: np.ndarray = cache["X_all"]  # type: ignore[assignment]
    y_all: np.ndarray = cache["y_all"]  # type: ignore[assignment]
    splits = cache["splits"]  # type: ignore[assignment]

    print(
        f"[1/4] Cache verified: {args.cache.name} | schema={cache['feature_schema']} | "
        f"train_det={splits['train']['detected']}/{splits['train']['total']} | "
        f"val_det={splits['validation']['detected']}/{splits['validation']['total']} | "
        f"test_det={splits['test']['detected']}/{splits['test']['total']} | leakage=0",
        flush=True,
    )

    handedness_summary = summarize_handedness(cache)
    for sp in ("train", "validation", "test"):
        st = handedness_summary["by_split"][sp]
        print(
            f"      Handedness {sp:10s}: Left={st['left_count']} ({st['left_pct']}%) | "
            f"Right={st['right_count']} ({st['right_pct']}%) | Other={st['other_count']}",
            flush=True,
        )

    # Extract split arrays
    tr_det = splits["train"]["det_mask"]
    va_det = splits["validation"]["det_mask"]
    te_det = splits["test"]["det_mask"]
    te_all = splits["test"]["mask"]

    X_tr_raw = X_all[tr_det]
    y_tr_raw = y_all[tr_det]
    X_tr_aug = np.vstack([X_tr_raw, flip_landmark_features_horizontal(X_tr_raw)]).astype(np.float32)
    y_tr_aug = np.concatenate([y_tr_raw, y_tr_raw]).astype(np.int32)

    X_va_orig = X_all[va_det]
    y_va_orig = y_all[va_det]
    X_va_mir = flip_landmark_features_horizontal(X_va_orig)
    val_total = int(splits["validation"]["total"])

    # Step 2: Validation grid search (STRICTLY on validation P8, P10)
    specs = build_candidate_specs()
    print(f"\n[2/4] Training & comparing {len(specs)} SVM candidates on validation (P8, P10)...", flush=True)
    candidate_rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for i, spec in enumerate(specs, start=1):
        cid = str(spec["candidate_id"])
        use_aug = bool(spec["augment_horizontal_flip"])
        X_fit = X_tr_aug if use_aug else X_tr_raw
        y_fit = y_tr_aug if use_aug else y_tr_raw

        pipe = build_svm_pipeline(spec, max_iter=args.max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            pipe.fit(X_fit, y_fit)
            fit_s = time.perf_counter() - t0

        pred_orig = np.asarray(pipe.predict(X_va_orig), dtype=np.int32)
        pred_mir = np.asarray(pipe.predict(X_va_mir), dtype=np.int32)

        m_orig = eval_predictions_on_split(y_va_orig, pred_orig, val_total, class_names)
        m_mir = eval_predictions_on_split(y_va_orig, pred_mir, val_total, class_names)

        comb_f1 = 0.5 * (float(m_orig["cond_macro_f1"]) + float(m_mir["cond_macro_f1"]))
        min_f1 = min(float(m_orig["cond_macro_f1"]), float(m_mir["cond_macro_f1"]))
        comb_acc = 0.5 * (float(m_orig["cond_accuracy"]) + float(m_mir["cond_accuracy"]))
        comb_watch = 0.5 * (float(m_orig["hard_watch_mean_recall"]) + float(m_mir["hard_watch_mean_recall"]))

        row = {
            "candidate_id": cid,
            "model_family": spec["model_family"],
            "kernel": spec["kernel"],
            "C": float(spec["C"]),
            "gamma": str(spec["gamma"]) if spec["gamma"] is not None else "n/a",
            "augment_horizontal_flip": use_aug,
            "train_samples_used": int(len(y_fit)),
            "fit_seconds": round(fit_s, 3),
            "convergence_warnings": len(caught),
            "val_orig_cond_accuracy": round(float(m_orig["cond_accuracy"]), 6),
            "val_orig_cond_macro_f1": round(float(m_orig["cond_macro_f1"]), 6),
            "val_orig_e2e_accuracy": round(float(m_orig["end_to_end_accuracy"]), 6),
            "val_mirrored_cond_accuracy": round(float(m_mir["cond_accuracy"]), 6),
            "val_mirrored_cond_macro_f1": round(float(m_mir["cond_macro_f1"]), 6),
            "val_mirrored_e2e_accuracy": round(float(m_mir["end_to_end_accuracy"]), 6),
            "combined_val_cond_macro_f1": round(comb_f1, 6),
            "min_val_cond_macro_f1": round(min_f1, 6),
            "combined_val_cond_accuracy": round(comb_acc, 6),
            "combined_hard_watch_recall": round(comb_watch, 6),
            "val_orig_hard_watch_recall": m_orig["hard_watch_recall"],
            "val_mirrored_hard_watch_recall": m_mir["hard_watch_recall"],
        }
        candidate_rows.append(row)
        fitted_models[cid] = pipe
        print(
            f"  [{i:2d}/{len(specs)}] {cid:26s} | fit={fit_s:6.2f}s | "
            f"val_orig: acc={row['val_orig_cond_accuracy']*100:5.2f}% f1={row['val_orig_cond_macro_f1']:.4f} e2e={row['val_orig_e2e_accuracy']*100:5.2f}% | "
            f"val_mir: acc={row['val_mirrored_cond_accuracy']*100:5.2f}% f1={row['val_mirrored_cond_macro_f1']:.4f} | "
            f"comb_f1={comb_f1:.4f}",
            flush=True,
        )

    # Rank strictly on validation (bilateral combined macro-F1 -> worst-hand macro-F1 -> orig macro-F1 -> comb acc -> smaller C -> faster fit)
    ranked = sorted(
        candidate_rows,
        key=lambda r: (
            -float(r["combined_val_cond_macro_f1"]),
            -float(r["min_val_cond_macro_f1"]),
            -float(r["val_orig_cond_macro_f1"]),
            -float(r["combined_val_cond_accuracy"]),
            float(r["C"]),
            float(r["fit_seconds"]),
        ),
    )
    winner = ranked[0]
    winner_id = str(winner["candidate_id"])
    winner_pipe = fitted_models[winner_id]

    print(
        f"\n  => SELECTED ON VALIDATION: {winner_id} "
        f"(val_orig_f1={winner['val_orig_cond_macro_f1']:.4f}, "
        f"val_mir_f1={winner['val_mirrored_cond_macro_f1']:.4f}, "
        f"combined_f1={winner['combined_val_cond_macro_f1']:.4f})",
        flush=True,
    )

    # Save winning model & metadata JSON
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "svm_landmark_asl_v1.joblib"
    metadata_path = args.model_dir / "svm_landmark_asl_v1_metadata.json"

    hyperparams = {
        "kernel": winner["kernel"],
        "C": float(winner["C"]),
        "gamma": (None if winner["gamma"] == "n/a" else (winner["gamma"] if winner["gamma"] == "scale" else float(winner["gamma"]))),
        "random_state": RANDOM_STATE,
        "probability": False,
        "decision_function_shape": "ovr" if winner["kernel"] == "rbf" else "ovr_linear_svc",
    }
    split_config = {
        "cache_path": str(Path(cache["cache_path"]).relative_to(PROJECT_ROOT).as_posix()),
        "cache_sha256": cache["cache_sha256"],
        "train_prefix_groups": splits["train"]["prefix_groups"],
        "validation_prefix_groups": splits["validation"]["prefix_groups"],
        "test_prefix_groups": splits["test"]["prefix_groups"],
        "train_detected_raw": int(splits["train"]["detected"]),
        "train_samples_fitted": int(winner["train_samples_used"]),
        "validation_detected": int(splits["validation"]["detected"]),
        "validation_total": int(splits["validation"]["total"]),
        "test_detected": int(splits["test"]["detected"]),
        "test_total": int(splits["test"]["total"]),
    }

    bundle = {
        "model": winner_pipe,
        "candidate_id": winner_id,
        "model_family": f"Pipeline(StandardScaler + {winner['model_family']})",
        "feature_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_count": LANDMARK_FEATURE_DIM,
        "feature_dim": LANDMARK_FEATURE_DIM,
        "mediapipe_version": cache["mediapipe_version"],
        "border_pad_ratio": float(cache["border_pad_ratio"]),
        "min_detection_confidence": float(cache["min_detection_confidence"]),
        "hand_landmarker_model": cache["hand_landmarker_model"],
        "class_names": np.asarray(class_names, dtype=str),
        "augment_horizontal_flip": bool(winner["augment_horizontal_flip"]),
        "hyperparameters": hyperparams,
        "split_config": split_config,
        "fit_seconds": float(winner["fit_seconds"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, model_path)
    model_sha256 = sha256_file(model_path)

    # Step 3: ONE-TIME final evaluation on unseen-group test (P2, P7) + bit-for-bit reload check
    print("\n[3/4] Running ONE-TIME final evaluation on unseen-group test (P2, P7)...", flush=True)
    X_te_det = X_all[te_det]
    y_te_det = y_all[te_det]
    y_te_all = y_all[te_all]
    te_det_flags = cache["detected"][te_all]  # type: ignore[index]

    test_summary, test_per_class_rows, orig_te_pred, orig_te_scores = evaluate_detailed_test(
        winner_pipe,
        X_te_det,
        y_te_det,
        y_te_all,
        te_det_flags,
        class_names,
    )

    # Bit-for-bit joblib reload verification
    reloaded_bundle = joblib.load(model_path)
    reloaded_pipe: Pipeline = reloaded_bundle["model"]
    reload_te_pred = np.asarray(reloaded_pipe.predict(X_te_det), dtype=np.int32)
    reload_te_scores = np.asarray(reloaded_pipe.decision_function(X_te_det), dtype=np.float64)
    pred_exact = bool(np.array_equal(orig_te_pred, reload_te_pred))
    scores_exact = bool(np.array_equal(orig_te_scores, reload_te_scores))
    if not (pred_exact and scores_exact):
        raise RuntimeError(
            f"Reloaded joblib model mismatch! pred_exact={pred_exact}, scores_exact={scores_exact}"
        )
    test_summary["joblib_reload_bitwise_parity"] = {
        "predict_exact_match": pred_exact,
        "decision_function_exact_match": scores_exact,
        "model_path": str(model_path.relative_to(PROJECT_ROOT).as_posix()),
        "model_sha256": model_sha256,
    }

    print(
        f"      Test (P2, P7) Conditional : acc={test_summary['cond_accuracy']*100:.2f}% | "
        f"macro_f1={test_summary['cond_macro_f1']:.4f} ({test_summary['correct_samples']}/{test_summary['detected_samples']})",
        flush=True,
    )
    print(
        f"      Test (P2, P7) End-to-End  : acc={test_summary['end_to_end_accuracy']*100:.2f}% "
        f"({test_summary['correct_samples']}/{test_summary['total_samples']}, counting {test_summary['undetected_samples']} misses as wrong)",
        flush=True,
    )
    print(
        f"      Test Mirrored (bilateral) : cond_acc={test_summary['test_mirrored_bilateral_check']['cond_accuracy']*100:.2f}% | "
        f"macro_f1={test_summary['test_mirrored_bilateral_check']['cond_macro_f1']:.4f}",
        flush=True,
    )
    print(f"      Joblib reload bitwise     : predict={pred_exact}, decision_function={scores_exact}", flush=True)

    # Write model metadata JSON
    metadata_doc = {
        "experiment": "direction_a_phase2_landmark_svm",
        "candidate_id": winner_id,
        "model_path": str(model_path.relative_to(PROJECT_ROOT).as_posix()),
        "model_sha256": model_sha256,
        "model_family": bundle["model_family"],
        "pipeline_steps": ["scaler:StandardScaler", f"clf:{type(winner_pipe.named_steps['clf']).__name__}"],
        "feature_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_dim": LANDMARK_FEATURE_DIM,
        "mediapipe_version": cache["mediapipe_version"],
        "border_pad_ratio": float(cache["border_pad_ratio"]),
        "min_detection_confidence": float(cache["min_detection_confidence"]),
        "hand_landmarker_model": cache["hand_landmarker_model"],
        "class_names": class_names,
        "class_count": len(class_names),
        "augment_horizontal_flip": bool(winner["augment_horizontal_flip"]),
        "hyperparameters": hyperparams,
        "split_config": split_config,
        "fit_seconds": float(winner["fit_seconds"]),
        "convergence_warnings": int(winner["convergence_warnings"]),
        "validation_selection_summary": {
            "selection_rule": (
                "Maximize combined_val_cond_macro_f1 = 0.5 * (val_orig_cond_macro_f1 + val_mirrored_cond_macro_f1) "
                "on validation (P8, P10), tie-breaking by min_val_cond_macro_f1, val_orig_cond_macro_f1, "
                "combined_val_cond_accuracy, smaller C, faster fit_seconds."
            ),
            "val_orig_cond_accuracy": winner["val_orig_cond_accuracy"],
            "val_orig_cond_macro_f1": winner["val_orig_cond_macro_f1"],
            "val_orig_e2e_accuracy": winner["val_orig_e2e_accuracy"],
            "val_mirrored_cond_accuracy": winner["val_mirrored_cond_accuracy"],
            "val_mirrored_cond_macro_f1": winner["val_mirrored_cond_macro_f1"],
            "val_mirrored_e2e_accuracy": winner["val_mirrored_e2e_accuracy"],
            "combined_val_cond_macro_f1": winner["combined_val_cond_macro_f1"],
        },
        "created_at": bundle["created_at"],
    }
    metadata_path.write_text(json.dumps(metadata_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 4: Fair side-by-side webcam benchmark (Landmark SVM vs HOG Method 3B)
    print("\n[4/4] Benchmarking Landmark SVM vs HOG Method 3B on real webcam image sets...", flush=True)
    webcam_report, webcam_csv_rows = run_webcam_benchmark(
        landmark_pipe=reloaded_pipe,
        class_names=class_names,
        hand_model_path=Path(cache["hand_landmarker_model"]),
        border_pad_ratio=float(cache["border_pad_ratio"]),
        min_detection_confidence=float(cache["min_detection_confidence"]),
    )
    for sid, sdata in webcam_report["sets"].items():  # type: ignore[union-attr]
        print(
            f"      {sid:32s} | det={sdata['landmark_detected_samples']:2d}/{sdata['total_samples']:2d} | "
            f"Landmark e2e={sdata['landmark_correct']:2d}/{sdata['total_samples']:2d} ({sdata['landmark_end_to_end_accuracy_pct']:5.2f}%) "
            f"cond={sdata['landmark_conditional_accuracy_pct']:5.2f}% | "
            f"HOG 3B e2e={sdata['hog_3b_correct']:2d}/{sdata['total_samples']:2d} ({sdata['hog_3b_end_to_end_accuracy_pct']:5.2f}%)",
            flush=True,
        )

    # Save all Phase 2 report artifacts (NEVER touching Phase 1 files!)
    handedness_path = args.report_dir / "phase2_handedness_summary.json"
    val_json_path = args.report_dir / "phase2_validation_comparison.json"
    val_csv_path = args.report_dir / "phase2_validation_comparison.csv"
    test_json_path = args.report_dir / "phase2_test_evaluation.json"
    test_csv_path = args.report_dir / "phase2_test_per_class.csv"
    webcam_json_path = args.report_dir / "phase2_webcam_benchmark.json"
    webcam_csv_path = args.report_dir / "phase2_webcam_benchmark_samples.csv"

    handedness_path.write_text(json.dumps(handedness_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    val_doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "validation_prefix_groups": ["P8", "P10"],
        "validation_total": val_total,
        "validation_detected": int(splits["validation"]["detected"]),
        "selection_rule": metadata_doc["validation_selection_summary"]["selection_rule"],
        "selected_candidate_id": winner_id,
        "selected_candidate": winner,
        "candidates_ranked": ranked,
    }
    val_json_path.write_text(json.dumps(val_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    val_csv_fields = [
        "rank",
        "candidate_id",
        "model_family",
        "kernel",
        "C",
        "gamma",
        "augment_horizontal_flip",
        "train_samples_used",
        "fit_seconds",
        "convergence_warnings",
        "val_orig_cond_accuracy",
        "val_orig_cond_macro_f1",
        "val_orig_e2e_accuracy",
        "val_mirrored_cond_accuracy",
        "val_mirrored_cond_macro_f1",
        "val_mirrored_e2e_accuracy",
        "combined_val_cond_macro_f1",
        "min_val_cond_macro_f1",
        "combined_val_cond_accuracy",
        "combined_hard_watch_recall",
    ]
    with val_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=val_csv_fields)
        writer.writeheader()
        for rank_idx, r in enumerate(ranked, start=1):
            row_out = {k: r[k] for k in val_csv_fields if k != "rank"}
            row_out["rank"] = rank_idx
            writer.writerow(row_out)

    test_json_path.write_text(json.dumps(test_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with test_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(test_per_class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(test_per_class_rows)

    webcam_json_path.write_text(json.dumps(webcam_report, indent=2, ensure_ascii=False), encoding="utf-8")
    with webcam_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(webcam_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(webcam_csv_rows)

    print("\nSaved Phase 2 artifacts:", flush=True)
    for p in (
        model_path,
        metadata_path,
        handedness_path,
        val_json_path,
        val_csv_path,
        test_json_path,
        test_csv_path,
        webcam_json_path,
        webcam_csv_path,
    ):
        print(f"  - {p.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    print("==========================================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
