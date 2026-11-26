"""Hướng A — Đợt 2b: Geometric C/D diagnosis + 86-dim Landmark Geometric v2 SVM experiment.

Reads directly from Phase 1 cache (`results/svm/experiments/landmark_asl_v1/asl_landmarks_cache.npz`)
WITHOUT re-running MediaPipe on 36,000 images.

Safety guarantees:
- NEVER touches any of the 7 pre-existing models in `models/svm/`.
- NEVER overwrites `models/svm/experiments/landmark_asl_v1/` or `results/svm/experiments/landmark_asl_v1/`.
- NEVER modifies/confirms webcam sample labels or deletes any images.
- Strictly selects v2 configuration ONLY on `validation (P8, P10)` (`val_orig` + `val_mirrored`).
- Saves winning v2 model to `models/svm/experiments/landmark_asl_v2/` and all reports to
  `results/svm/experiments/landmark_asl_v2/`.
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

from scripts.train_asl_landmark_svm_phase2 import (
    DEFAULT_CACHE_PATH,
    collect_webcam_sample_sets,
    load_and_verify_cache,
    predict_landmark_top_k,
    sha256_array,
    sha256_file,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.model.train_svm import RANDOM_STATE
from src.preprocess.webcam_features import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    canonicalize_saved_crop_bgr,
    create_hand_landmarker,
    extract_landmark_geometric_features_v2,
    extract_mediapipe_landmarks_feature_from_bgr,
    flip_landmark_features_horizontal,
    rotate_landmark_features_2d,
)

V1_MODEL_PATH = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v1" / "svm_landmark_asl_v1.joblib"
V1_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1"
V2_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v2"
V2_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v2"

VAL_WATCH_CLASSES = ["D", "C", "0", "O", "M", "N", "S", "T", "U", "R", "2", "V", "6", "W", "1"]
TEST_SPOTLIGHT_CLASSES = ["0", "P", "Q", "O", "G", "C", "D", "M", "N", "S", "R", "U"]
DIAG_CLASSES = ["C", "D", "0", "1", "O"]


def compute_geometry_metrics_for_xy(X_xy_42: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute interpretable 2D geometric summary stats from (N, 42) normalized landmarks."""
    pts = np.asarray(X_xy_42, dtype=np.float32).reshape(-1, 21, 2)

    def dist(a: int, b: int) -> np.ndarray:
        return np.linalg.norm(pts[:, a, :] - pts[:, b, :], axis=-1)

    def cos_at(a: int, b: int, c: int) -> np.ndarray:
        u = pts[:, a, :] - pts[:, b, :]
        v = pts[:, c, :] - pts[:, b, :]
        dot = np.sum(u * v, axis=-1)
        denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        return np.where(denom > 1e-8, np.clip(dot / np.maximum(denom, 1e-8), -1.0, 1.0), 0.0)

    # Palm axis from wrist (0) to middle MCP (9): angle from upright (0, -1) in degrees
    axis = pts[:, 9, :] - pts[:, 0, :]
    tilt_deg = np.rad2deg(np.arctan2(axis[:, 0], -axis[:, 1]))

    raw = {
        "d_thumb4_index8": dist(4, 8),
        "d_thumb4_middle12": dist(4, 12),
        "d_thumb4_middle_pip10": dist(4, 10),
        "d_thumb4_wrist0": dist(4, 0),
        "d_index8_wrist0": dist(8, 0),
        "d_middle12_wrist0": dist(12, 0),
        "d_ring16_wrist0": dist(16, 0),
        "d_pinky20_wrist0": dist(20, 0),
        "d_index8_mcp5": dist(8, 5),
        "d_middle12_mcp9": dist(12, 9),
        "d_ring16_mcp13": dist(16, 13),
        "d_pinky20_mcp17": dist(20, 17),
        "cos_index_pip6": cos_at(5, 6, 7),
        "cos_middle_pip10": cos_at(9, 10, 11),
        "abs_palm_tilt_deg": np.abs(tilt_deg),
        "signed_palm_tilt_deg": tilt_deg,
    }
    stats: dict[str, dict[str, float]] = {}
    for k, vals in raw.items():
        stats[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "p10": round(float(np.percentile(vals, 10)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4),
        }
    return stats


def extract_webcam_xy_cache(
    datasets: list[dict[str, object]],
    hand_model_path: Path,
    border_pad_ratio: float,
    min_detection_confidence: float,
) -> dict[str, dict[str, object]]:
    """Run MediaPipe once per unique webcam folder (read-only) and cache canonical crops & 42-dim xy."""
    detector = create_hand_landmarker(
        model_path=hand_model_path,
        min_detection_confidence=min_detection_confidence,
        num_hands=1,
    )
    folder_cache: dict[str, dict[str, object]] = {}
    try:
        for ds in datasets:
            for sdir, meta, true_label in ds["items"]:  # type: ignore[union-attr]
                key = str(sdir.resolve())
                if key in folder_cache:
                    continue
                crop_path = sdir / "received_crop.png"
                crop_bgr = cv2.imread(str(crop_path))
                if crop_bgr is None:
                    raise ValueError(f"Cannot read crop: {crop_path}")
                feat_42, mp_info = extract_mediapipe_landmarks_feature_from_bgr(
                    crop_bgr,
                    detector=detector,
                    metadata=meta,
                    flip_horizontal=False,
                    border_pad_ratio=border_pad_ratio,
                )
                folder_cache[key] = {
                    "sdir": sdir,
                    "folder_name": sdir.name,
                    "meta": meta,
                    "true_label": true_label,
                    "crop_bgr": crop_bgr,
                    "canonical_bgr": canonicalize_saved_crop_bgr(crop_bgr, metadata=meta),
                    "hand_detected": feat_42 is not None,
                    "feat_xy_42": feat_42,
                    "feat_v2_86": extract_landmark_geometric_features_v2(feat_42) if feat_42 is not None else None,
                    "mediapipe_handedness": mp_info.get("mediapipe_handedness") or "",
                }
    finally:
        detector.close()
    return folder_cache


def run_cd_geometry_diagnosis(
    cache: dict[str, object],
    datasets: list[dict[str, object]],
    webcam_cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Read-only geometric comparison of asl_processed train (C, D, 0, 1, O) vs webcam (C, D)."""
    X_all: np.ndarray = cache["X_all"]  # type: ignore[assignment]
    y_all: np.ndarray = cache["y_all"]  # type: ignore[assignment]
    class_names: list[str] = cache["class_names"]  # type: ignore[assignment]
    tr_det = cache["splits"]["train"]["det_mask"]  # type: ignore[index]

    X_tr = X_all[tr_det]
    y_tr = y_all[tr_det]

    # ASL train stats & mirror-invariant centroids in 42-dim xy AND 44-dim pure geometry
    asl_stats: dict[str, object] = {}
    asl_xy_centroids: dict[str, np.ndarray] = {}
    asl_geom_centroids: dict[str, np.ndarray] = {}
    for cls in DIAG_CLASSES:
        cid = class_names.index(cls)
        Xc = X_tr[y_tr == cid]
        Xc_v2 = extract_landmark_geometric_features_v2(Xc)
        asl_xy_centroids[cls] = Xc.mean(axis=0)
        asl_geom_centroids[cls] = Xc_v2[:, LANDMARK_FEATURE_DIM:].mean(axis=0)
        asl_stats[cls] = {
            "train_detected_count": int(len(Xc)),
            "metrics": compute_geometry_metrics_for_xy(Xc),
        }

    # Gather webcam C and D from pilot_15 and confirmed_32
    ds_map = {str(d["set_id"]): d for d in datasets}
    webcam_groups: dict[str, object] = {}
    for set_id in ("webcam_pilot_15", "webcam_diagnostics_confirmed_32"):
        for target_cls in ("C", "D"):
            feats_xy = []
            for sdir, _meta, lbl in ds_map[set_id]["items"]:  # type: ignore[union-attr]
                if lbl != target_cls:
                    continue
                rec = webcam_cache[str(sdir.resolve())]
                if rec["hand_detected"]:
                    feats_xy.append(rec["feat_xy_42"])
            if not feats_xy:
                continue
            arr_xy = np.asarray(feats_xy, dtype=np.float32)
            arr_xy_flip = flip_landmark_features_horizontal(arr_xy)
            arr_geom = extract_landmark_geometric_features_v2(arr_xy)[:, LANDMARK_FEATURE_DIM:]

            # Compare mean distance to ASL train centroids (using best of orig/flip for xy so left-hand aligns)
            xy_dist_to_asl = {}
            geom_dist_to_asl = {}
            for ref_cls in DIAG_CLASSES:
                mu_xy = asl_xy_centroids[ref_cls]
                d_orig = np.linalg.norm(arr_xy - mu_xy, axis=1)
                d_flip = np.linalg.norm(arr_xy_flip - mu_xy, axis=1)
                xy_dist_to_asl[ref_cls] = round(float(np.mean(np.minimum(d_orig, d_flip))), 4)

                mu_g = asl_geom_centroids[ref_cls]
                geom_dist_to_asl[ref_cls] = round(float(np.mean(np.linalg.norm(arr_geom - mu_g, axis=1))), 4)

            webcam_groups[f"{set_id}__{target_cls}"] = {
                "set_id": set_id,
                "true_label": target_cls,
                "detected_count": int(len(arr_xy)),
                "metrics": compute_geometry_metrics_for_xy(arr_xy),
                "mean_mirror_aligned_xy_dist_to_asl_train_centroid": dict(
                    sorted(xy_dist_to_asl.items(), key=lambda kv: kv[1])
                ),
                "mean_44dim_geom_dist_to_asl_train_centroid": dict(
                    sorted(geom_dist_to_asl.items(), key=lambda kv: kv[1])
                ),
            }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asl_processed_train_clusters": asl_stats,
        "webcam_cd_clusters": webcam_groups,
    }


def build_v2_candidate_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    # 1. Horizontal flip only on train (2x = 41,606)
    for c_val in (0.1, 1.0):
        specs.append(
            {
                "candidate_id": f"v2_linear_C{c_val:g}_flip",
                "model_family": "LinearSVC",
                "kernel": "linear",
                "C": float(c_val),
                "gamma": None,
                "aug_mode": "flip",
                "rotation_angles_deg": [0.0],
            }
        )
    for c_val in (1.0, 10.0, 30.0):
        for gamma in ("scale", 0.005, 0.025):
            specs.append(
                {
                    "candidate_id": f"v2_rbf_C{c_val:g}_g{gamma}_flip",
                    "model_family": "SVC(rbf)",
                    "kernel": "rbf",
                    "C": float(c_val),
                    "gamma": gamma,
                    "aug_mode": "flip",
                    "rotation_angles_deg": [0.0],
                }
            )
    # 2. Horizontal flip + light in-plane rotation +-12 deg on train (6x = 124,818)
    specs.append(
        {
            "candidate_id": "v2_linear_C0.1_flip_rot12",
            "model_family": "LinearSVC",
            "kernel": "linear",
            "C": 0.1,
            "gamma": None,
            "aug_mode": "flip_rot12",
            "rotation_angles_deg": [-12.0, 0.0, 12.0],
        }
    )
    for c_val in (1.0, 10.0, 30.0):
        for gamma in ("scale", 0.025):
            specs.append(
                {
                    "candidate_id": f"v2_rbf_C{c_val:g}_g{gamma}_flip_rot12",
                    "model_family": "SVC(rbf)",
                    "kernel": "rbf",
                    "C": float(c_val),
                    "gamma": gamma,
                    "aug_mode": "flip_rot12",
                    "rotation_angles_deg": [-12.0, 0.0, 12.0],
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


def eval_on_val(
    y_true_det: np.ndarray,
    y_pred_det: np.ndarray,
    total_count: int,
    class_names: list[str],
) -> dict[str, object]:
    n_det = int(len(y_true_det))
    n_corr = int(np.sum(y_true_det == y_pred_det))
    cond_acc = float(n_corr / n_det)
    cond_f1 = float(
        f1_score(y_true_det, y_pred_det, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    e2e_acc = float(n_corr / total_count)
    _, rec, _, _ = precision_recall_fscore_support(
        y_true_det, y_pred_det, labels=list(range(len(class_names))), zero_division=0
    )
    watch = {c: round(float(rec[class_names.index(c)]), 4) for c in VAL_WATCH_CLASSES}
    return {
        "cond_accuracy": cond_acc,
        "cond_macro_f1": cond_f1,
        "e2e_accuracy": e2e_acc,
        "watch_recall": watch,
        "watch_mean_recall": float(np.mean(list(watch.values()))),
    }


def evaluate_test_once(
    pipe: Pipeline,
    X_te_v2_orig: np.ndarray,
    X_te_v2_mir: np.ndarray,
    y_te_det: np.ndarray,
    y_te_all: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    y_pred = np.asarray(pipe.predict(X_te_v2_orig), dtype=np.int32)
    scores = np.asarray(pipe.decision_function(X_te_v2_orig), dtype=np.float64)
    infer_s = time.perf_counter() - t0

    n_total = int(len(y_te_all))
    n_det = int(len(y_te_det))
    n_undet = int(n_total - n_det)
    n_corr = int(np.sum(y_te_det == y_pred))
    cond_acc = float(accuracy_score(y_te_det, y_pred))
    cond_macro_f1 = float(
        f1_score(y_te_det, y_pred, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    e2e_acc = float(n_corr / n_total)

    y_pred_mir = np.asarray(pipe.predict(X_te_v2_mir), dtype=np.int32)
    mir_corr = int(np.sum(y_te_det == y_pred_mir))
    mir_cond_acc = float(mir_corr / n_det)
    mir_macro_f1 = float(
        f1_score(y_te_det, y_pred_mir, labels=list(range(len(class_names))), average="macro", zero_division=0)
    )
    mir_e2e_acc = float(mir_corr / n_total)

    prec, rec, f1s, sup_det = precision_recall_fscore_support(
        y_te_det, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class_rows: list[dict[str, object]] = []
    for i, cname in enumerate(class_names):
        tot_c = int(np.sum(y_te_all == i))
        det_c = int(sup_det[i])
        corr_c = int(np.sum((y_te_det == i) & (y_pred == i)))
        per_class_rows.append(
            {
                "class_index": i,
                "class_name": cname,
                "support_total": tot_c,
                "support_detected": det_c,
                "undetected_count": int(tot_c - det_c),
                "detection_rate": round(det_c / tot_c, 6) if tot_c else 0.0,
                "correct_count": corr_c,
                "cond_precision": round(float(prec[i]), 6),
                "cond_recall": round(float(rec[i]), 6),
                "cond_f1": round(float(f1s[i]), 6),
                "end_to_end_recall": round(corr_c / tot_c, 6) if tot_c else 0.0,
            }
        )

    cm = confusion_matrix(y_te_det, y_pred, labels=list(range(len(class_names))))
    confusions: list[dict[str, object]] = []
    for ti in range(len(class_names)):
        row_tot = int(cm[ti].sum())
        for pi in range(len(class_names)):
            if ti != pi and cm[ti, pi] > 0:
                cnt = int(cm[ti, pi])
                confusions.append(
                    {
                        "true_label": class_names[ti],
                        "predicted_label": class_names[pi],
                        "count": cnt,
                        "pct_of_true_detected": round(cnt / row_tot * 100.0, 2) if row_tot else 0.0,
                    }
                )
    confusions.sort(key=lambda r: (-int(r["count"]), str(r["true_label"]), str(r["predicted_label"])))

    spotlight = {r["class_name"]: r for r in per_class_rows if r["class_name"] in TEST_SPOTLIGHT_CLASSES}
    golden_n = min(64, n_det)
    summary = {
        "split": "test",
        "prefix_groups": ["P2", "P7"],
        "total_samples": n_total,
        "detected_samples": n_det,
        "undetected_samples": n_undet,
        "detection_rate": round(n_det / n_total, 6),
        "correct_samples": n_corr,
        "cond_accuracy": round(cond_acc, 6),
        "cond_macro_f1": round(cond_macro_f1, 6),
        "end_to_end_accuracy": round(e2e_acc, 6),
        "inference_seconds_all_detected": round(infer_s, 4),
        "inference_ms_per_detected_sample": round(infer_s / n_det * 1000.0, 4) if n_det else 0.0,
        "test_mirrored_bilateral_check": {
            "correct_samples": mir_corr,
            "cond_accuracy": round(mir_cond_acc, 6),
            "cond_macro_f1": round(mir_macro_f1, 6),
            "end_to_end_accuracy": round(mir_e2e_acc, 6),
        },
        "top_10_confusions": confusions[:10],
        "spotlight_classes": spotlight,
        "weakest_10_by_cond_f1": sorted(per_class_rows, key=lambda r: (float(r["cond_f1"]), float(r["end_to_end_recall"])))[:10],
        "per_class": per_class_rows,
        "golden_reload_verification": {
            "slice_size": golden_n,
            "y_pred_first_64": y_pred[:golden_n].tolist(),
            "decision_scores_first_64_sha256": sha256_array(scores[:golden_n]),
            "full_y_pred_sha256": sha256_array(y_pred),
            "full_decision_scores_sha256": sha256_array(scores),
        },
    }
    return summary, per_class_rows, y_pred, scores


def run_3way_webcam_benchmark(
    v2_pipe: Pipeline,
    v1_pipe: Pipeline,
    class_names: list[str],
    datasets: list[dict[str, object]],
    excluded_info: dict[str, object],
    webcam_cache: dict[str, dict[str, object]],
    border_pad_ratio: float,
    min_detection_confidence: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compare Landmark SVM v2 vs Landmark SVM v1 vs HOG Method 3B on exact same webcam crops."""
    hog_predictor = SVMSignPredictor(model_path=PROJECT_ROOT / DEFAULT_MODEL_PATH)
    per_folder_eval: dict[str, dict[str, object]] = {}

    for key, rec in webcam_cache.items():
        true_label = str(rec["true_label"])
        meta = rec["meta"]
        hand_det = bool(rec["hand_detected"])

        if hand_det:
            v2_res = predict_landmark_top_k(v2_pipe, rec["feat_v2_86"], class_names, top_k=3)
            v1_res = predict_landmark_top_k(v1_pipe, rec["feat_xy_42"], class_names, top_k=3)
            v2_pred = str(v2_res["predicted_label"])
            v2_top3 = v2_res["top_k"]
            v1_pred = str(v1_res["predicted_label"])
            v1_top3 = v1_res["top_k"]
        else:
            v2_pred = "__NO_HAND__"
            v2_top3 = []
            v1_pred = "__NO_HAND__"
            v1_top3 = []

        hog_res = hog_predictor.predict_image_bgr(
            rec["canonical_bgr"],
            top_k=3,
            include_diagnostics=False,
            experimental_webcam_mode=True,
            already_flipped=False,
        )
        hog_pred = str(hog_res["predicted_label"])

        per_folder_eval[key] = {
            "folder_name": rec["folder_name"],
            "true_label": true_label,
            "participant_id": meta.get("participant_id") or "",
            "gesture_instance_id": meta.get("gesture_instance_id") or "",
            "hand_detected": hand_det,
            "mediapipe_handedness": rec["mediapipe_handedness"],
            "v2_pred": v2_pred,
            "v2_correct": (hand_det and v2_pred == true_label),
            "v2_top3": v2_top3,
            "v1_pred": v1_pred,
            "v1_correct": (hand_det and v1_pred == true_label),
            "v1_top3": v1_top3,
            "hog_3b_pred": hog_pred,
            "hog_3b_correct": (hog_pred == true_label),
        }

    sets_report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []

    for ds in datasets:
        set_id = str(ds["set_id"])
        sample_rows = []
        for idx, (sdir, _meta, _lbl) in enumerate(ds["items"], start=1):  # type: ignore[union-attr]
            r = dict(per_folder_eval[str(sdir.resolve())])
            r["stt"] = idx
            r["set_id"] = set_id
            sample_rows.append(r)
            csv_rows.append(
                {
                    "set_id": set_id,
                    "stt": idx,
                    "folder_name": r["folder_name"],
                    "true_label": r["true_label"],
                    "gesture_instance_id": r["gesture_instance_id"],
                    "hand_detected": r["hand_detected"],
                    "v2_pred": r["v2_pred"],
                    "v2_correct": r["v2_correct"],
                    "v2_top3": "; ".join(f"{t['label']}:{t['decision_score']}" for t in r["v2_top3"]),
                    "v1_pred": r["v1_pred"],
                    "v1_correct": r["v1_correct"],
                    "hog_3b_pred": r["hog_3b_pred"],
                    "hog_3b_correct": r["hog_3b_correct"],
                }
            )

        n_tot = len(sample_rows)
        det_rows = [r for r in sample_rows if r["hand_detected"]]
        n_det = len(det_rows)
        v2_ok = sum(1 for r in sample_rows if r["v2_correct"])
        v1_ok = sum(1 for r in sample_rows if r["v1_correct"])
        hog_ok = sum(1 for r in sample_rows if r["hog_3b_correct"])
        hog_ok_det = sum(1 for r in det_rows if r["hog_3b_correct"])

        per_class = {}
        for cls in sorted({str(r["true_label"]) for r in sample_rows}):
            cr = [r for r in sample_rows if r["true_label"] == cls]
            cd = [r for r in cr if r["hand_detected"]]
            c_v2 = sum(1 for r in cr if r["v2_correct"])
            c_v1 = sum(1 for r in cr if r["v1_correct"])
            c_hog = sum(1 for r in cr if r["hog_3b_correct"])
            per_class[cls] = {
                "total": len(cr),
                "detected": len(cd),
                "v2_correct": c_v2,
                "v2_e2e_pct": round(c_v2 / len(cr) * 100.0, 2),
                "v2_cond_pct": round(c_v2 / len(cd) * 100.0, 2) if cd else 0.0,
                "v2_pred_dist": dict(Counter(str(r["v2_pred"]) for r in cr)),
                "v1_correct": c_v1,
                "v1_e2e_pct": round(c_v1 / len(cr) * 100.0, 2),
                "v1_cond_pct": round(c_v1 / len(cd) * 100.0, 2) if cd else 0.0,
                "v1_pred_dist": dict(Counter(str(r["v1_pred"]) for r in cr)),
                "hog_3b_correct": c_hog,
                "hog_3b_e2e_pct": round(c_hog / len(cr) * 100.0, 2),
                "hog_3b_pred_dist": dict(Counter(str(r["hog_3b_pred"]) for r in cr)),
            }

        sets_report[set_id] = {
            "title": ds["title"],
            "label_confirmed_status": ds["label_confirmed_status"],
            "total_samples": n_tot,
            "detected_samples": n_det,
            "detection_rate_pct": round(n_det / n_tot * 100.0, 2) if n_tot else 0.0,
            "v2_correct": v2_ok,
            "v2_e2e_pct": round(v2_ok / n_tot * 100.0, 2) if n_tot else 0.0,
            "v2_cond_pct": round(v2_ok / n_det * 100.0, 2) if n_det else 0.0,
            "v1_correct": v1_ok,
            "v1_e2e_pct": round(v1_ok / n_tot * 100.0, 2) if n_tot else 0.0,
            "v1_cond_pct": round(v1_ok / n_det * 100.0, 2) if n_det else 0.0,
            "hog_3b_correct": hog_ok,
            "hog_3b_e2e_pct": round(hog_ok / n_tot * 100.0, 2) if n_tot else 0.0,
            "hog_3b_on_det_pct": round(hog_ok_det / n_det * 100.0, 2) if n_det else 0.0,
            "per_class": per_class,
            "v2_errors_or_misses": [
                {
                    "stt": r["stt"],
                    "folder_name": r["folder_name"],
                    "gesture_instance_id": r["gesture_instance_id"],
                    "true_label": r["true_label"],
                    "hand_detected": r["hand_detected"],
                    "v2_pred": r["v2_pred"],
                    "v2_top3": r["v2_top3"],
                    "v1_pred": r["v1_pred"],
                    "hog_3b_pred": r["hog_3b_pred"],
                }
                for r in sample_rows
                if not r["v2_correct"]
            ],
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "v2_feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
        "v2_feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
        "v1_feature_schema": LANDMARK_FEATURE_SCHEMA,
        "v1_feature_dim": LANDMARK_FEATURE_DIM,
        "border_pad_ratio": float(border_pad_ratio),
        "min_detection_confidence": float(min_detection_confidence),
        "hog_baseline_model": str(DEFAULT_MODEL_PATH.as_posix()),
        "excluded_unconfirmed_sets": excluded_info,
        "sets": sets_report,
    }
    return report, csv_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    p.add_argument("--v1-model", type=Path, default=V1_MODEL_PATH)
    p.add_argument("--model-dir", type=Path, default=V2_MODEL_DIR)
    p.add_argument("--report-dir", type=Path, default=V2_REPORT_DIR)
    p.add_argument("--max-iter", type=int, default=10_000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    print("================ HƯỚNG A — ĐỢT 2B: GEOMETRIC V2 LANDMARK SVM ================", flush=True)
    cache = load_and_verify_cache(args.cache)
    class_names: list[str] = cache["class_names"]  # type: ignore[assignment]
    X_all: np.ndarray = cache["X_all"]  # type: ignore[assignment]
    y_all: np.ndarray = cache["y_all"]  # type: ignore[assignment]
    splits = cache["splits"]  # type: ignore[assignment]

    datasets, excluded_info = collect_webcam_sample_sets()
    webcam_cache = extract_webcam_xy_cache(
        datasets=datasets,
        hand_model_path=Path(cache["hand_landmarker_model"]),
        border_pad_ratio=float(cache["border_pad_ratio"]),
        min_detection_confidence=float(cache["min_detection_confidence"]),
    )

    # Step 1: Read-only geometric C/D diagnosis
    print("[1/4] Running read-only C/D geometry diagnosis (ASL train vs webcam C/D)...", flush=True)
    cd_diag = run_cd_geometry_diagnosis(cache, datasets, webcam_cache)
    for k, g in cd_diag["webcam_cd_clusters"].items():  # type: ignore[union-attr]
        m = g["metrics"]
        print(
            f"      {k:36s} (n={g['detected_count']:2d}) | d(4,8)={m['d_thumb4_index8']['mean']:.3f} "
            f"d(4,12)={m['d_thumb4_middle12']['mean']:.3f} d(8,0)={m['d_index8_wrist0']['mean']:.3f} "
            f"|tilt|={m['abs_palm_tilt_deg']['mean']:.1f}deg | nearest_xy={list(g['mean_mirror_aligned_xy_dist_to_asl_train_centroid'].items())[:3]} "
            f"| nearest_geom={list(g['mean_44dim_geom_dist_to_asl_train_centroid'].items())[:3]}",
            flush=True,
        )

    # Step 2: Prepare v2 features (strictly transform 21 xy BEFORE computing v2, augment ONLY train)
    tr_det = splits["train"]["det_mask"]
    va_det = splits["validation"]["det_mask"]
    te_det = splits["test"]["det_mask"]
    te_all = splits["test"]["mask"]

    X_tr_xy = X_all[tr_det]
    y_tr = y_all[tr_det]

    # Precompute flip-only (2x) and flip+rot12 (6x) train matrices once
    X_tr_flip_xy = np.vstack([X_tr_xy, flip_landmark_features_horizontal(X_tr_xy)])
    X_tr_v2_flip = extract_landmark_geometric_features_v2(X_tr_flip_xy)
    y_tr_flip = np.concatenate([y_tr, y_tr]).astype(np.int32)

    rot_xy_blocks = []
    rot_y_blocks = []
    for ang in (-12.0, 0.0, 12.0):
        base_rot = X_tr_xy if ang == 0.0 else rotate_landmark_features_2d(X_tr_xy, ang)
        flip_rot = flip_landmark_features_horizontal(base_rot)
        rot_xy_blocks.extend([base_rot, flip_rot])
        rot_y_blocks.extend([y_tr, y_tr])
    X_tr_v2_rot12 = extract_landmark_geometric_features_v2(np.vstack(rot_xy_blocks))
    y_tr_rot12 = np.concatenate(rot_y_blocks).astype(np.int32)

    # Validation orig & mirrored (NEVER augmented into train)
    X_va_xy = X_all[va_det]
    y_va = y_all[va_det]
    X_va_v2_orig = extract_landmark_geometric_features_v2(X_va_xy)
    X_va_v2_mir = extract_landmark_geometric_features_v2(flip_landmark_features_horizontal(X_va_xy))
    val_total = int(splits["validation"]["total"])

    # Load v1 validation & test baselines for direct reporting
    v1_val_doc = json.loads((V1_REPORT_DIR / "phase2_validation_comparison.json").read_text(encoding="utf-8"))
    v1_test_doc = json.loads((V1_REPORT_DIR / "phase2_test_evaluation.json").read_text(encoding="utf-8"))
    v1_val_comb_f1 = float(v1_val_doc["selected_candidate"]["combined_val_cond_macro_f1"])

    specs = build_v2_candidate_specs()
    print(
        f"\n[2/4] Comparing {len(specs)} v2 candidates (86-dim) strictly on validation (P8, P10) "
        f"[v1 baseline combined_f1={v1_val_comb_f1:.4f}]...",
        flush=True,
    )
    cand_rows: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for i, spec in enumerate(specs, start=1):
        cid = str(spec["candidate_id"])
        X_fit = X_tr_v2_rot12 if spec["aug_mode"] == "flip_rot12" else X_tr_v2_flip
        y_fit = y_tr_rot12 if spec["aug_mode"] == "flip_rot12" else y_tr_flip

        pipe = build_svm_pipeline(spec, max_iter=args.max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            pipe.fit(X_fit, y_fit)
            fit_s = time.perf_counter() - t0

        pred_orig = np.asarray(pipe.predict(X_va_v2_orig), dtype=np.int32)
        pred_mir = np.asarray(pipe.predict(X_va_v2_mir), dtype=np.int32)
        mo = eval_on_val(y_va, pred_orig, val_total, class_names)
        mm = eval_on_val(y_va, pred_mir, val_total, class_names)
        comb_f1 = 0.5 * (float(mo["cond_macro_f1"]) + float(mm["cond_macro_f1"]))
        min_f1 = min(float(mo["cond_macro_f1"]), float(mm["cond_macro_f1"]))
        comb_acc = 0.5 * (float(mo["cond_accuracy"]) + float(mm["cond_accuracy"]))
        comb_watch = 0.5 * (float(mo["watch_mean_recall"]) + float(mm["watch_mean_recall"]))

        row = {
            "candidate_id": cid,
            "model_family": spec["model_family"],
            "kernel": spec["kernel"],
            "C": float(spec["C"]),
            "gamma": str(spec["gamma"]) if spec["gamma"] is not None else "n/a",
            "aug_mode": spec["aug_mode"],
            "rotation_angles_deg": spec["rotation_angles_deg"],
            "train_samples_used": int(len(y_fit)),
            "fit_seconds": round(fit_s, 3),
            "convergence_warnings": len(caught),
            "val_orig_cond_accuracy": round(float(mo["cond_accuracy"]), 6),
            "val_orig_cond_macro_f1": round(float(mo["cond_macro_f1"]), 6),
            "val_orig_e2e_accuracy": round(float(mo["e2e_accuracy"]), 6),
            "val_mirrored_cond_accuracy": round(float(mm["cond_accuracy"]), 6),
            "val_mirrored_cond_macro_f1": round(float(mm["cond_macro_f1"]), 6),
            "val_mirrored_e2e_accuracy": round(float(mm["e2e_accuracy"]), 6),
            "combined_val_cond_macro_f1": round(comb_f1, 6),
            "delta_vs_v1_combined_f1": round(comb_f1 - v1_val_comb_f1, 6),
            "min_val_cond_macro_f1": round(min_f1, 6),
            "combined_val_cond_accuracy": round(comb_acc, 6),
            "combined_watch_recall": round(comb_watch, 6),
            "val_orig_watch_recall": mo["watch_recall"],
            "val_mirrored_watch_recall": mm["watch_recall"],
        }
        cand_rows.append(row)
        fitted_models[cid] = pipe
        print(
            f"  [{i:2d}/{len(specs)}] {cid:28s} | fit={fit_s:6.2f}s | "
            f"orig: acc={row['val_orig_cond_accuracy']*100:5.2f}% f1={row['val_orig_cond_macro_f1']:.4f} e2e={row['val_orig_e2e_accuracy']*100:5.2f}% | "
            f"mir: f1={row['val_mirrored_cond_macro_f1']:.4f} | comb_f1={comb_f1:.4f} ({comb_f1 - v1_val_comb_f1:+.4f})",
            flush=True,
        )

    ranked = sorted(
        cand_rows,
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
        f"\n  => SELECTED v2 ON VALIDATION: {winner_id} "
        f"(comb_f1={winner['combined_val_cond_macro_f1']:.4f}, vs v1 {winner['delta_vs_v1_combined_f1']:+.4f})",
        flush=True,
    )

    # Save v2 model & metadata in SEPARATE landmark_asl_v2 folder
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "svm_landmark_asl_v2.joblib"
    metadata_path = args.model_dir / "svm_landmark_asl_v2_metadata.json"

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
        "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
        "base_xy_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_count": LANDMARK_GEOM_V2_FEATURE_DIM,
        "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
        "mediapipe_version": cache["mediapipe_version"],
        "border_pad_ratio": float(cache["border_pad_ratio"]),
        "min_detection_confidence": float(cache["min_detection_confidence"]),
        "hand_landmarker_model": cache["hand_landmarker_model"],
        "class_names": np.asarray(class_names, dtype=str),
        "aug_mode": winner["aug_mode"],
        "rotation_angles_deg": winner["rotation_angles_deg"],
        "augment_horizontal_flip": True,
        "hyperparameters": hyperparams,
        "split_config": split_config,
        "fit_seconds": float(winner["fit_seconds"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, model_path)
    model_sha256 = sha256_file(model_path)

    # Step 3: ONE-TIME final test evaluation on unseen-group test (P2, P7) + bitwise joblib reload check
    print("\n[3/4] Running ONE-TIME final evaluation on unseen-group test (P2, P7)...", flush=True)
    X_te_xy = X_all[te_det]
    y_te_det = y_all[te_det]
    y_te_all = y_all[te_all]
    X_te_v2_orig = extract_landmark_geometric_features_v2(X_te_xy)
    X_te_v2_mir = extract_landmark_geometric_features_v2(flip_landmark_features_horizontal(X_te_xy))

    test_summary, test_per_class_rows, orig_pred, orig_scores = evaluate_test_once(
        winner_pipe, X_te_v2_orig, X_te_v2_mir, y_te_det, y_te_all, class_names
    )

    reloaded_v2_bundle = joblib.load(model_path)
    reloaded_v2_pipe: Pipeline = reloaded_v2_bundle["model"]
    reload_pred = np.asarray(reloaded_v2_pipe.predict(X_te_v2_orig), dtype=np.int32)
    reload_scores = np.asarray(reloaded_v2_pipe.decision_function(X_te_v2_orig), dtype=np.float64)
    assert np.array_equal(orig_pred, reload_pred), "v2 reloaded predict mismatch!"
    assert np.array_equal(orig_scores, reload_scores), "v2 reloaded decision_function mismatch!"

    test_summary["joblib_reload_bitwise_parity"] = {
        "predict_exact_match": True,
        "decision_function_exact_match": True,
        "model_path": str(model_path.relative_to(PROJECT_ROOT).as_posix()),
        "model_sha256": model_sha256,
    }
    test_summary["comparison_vs_v1"] = {
        "v1_cond_accuracy": v1_test_doc["cond_accuracy"],
        "v1_cond_macro_f1": v1_test_doc["cond_macro_f1"],
        "v1_end_to_end_accuracy": v1_test_doc["end_to_end_accuracy"],
        "delta_cond_accuracy": round(float(test_summary["cond_accuracy"]) - float(v1_test_doc["cond_accuracy"]), 6),
        "delta_cond_macro_f1": round(float(test_summary["cond_macro_f1"]) - float(v1_test_doc["cond_macro_f1"]), 6),
        "delta_end_to_end_accuracy": round(
            float(test_summary["end_to_end_accuracy"]) - float(v1_test_doc["end_to_end_accuracy"]), 6
        ),
    }
    print(
        f"      Test v2 Cond  : acc={test_summary['cond_accuracy']*100:.2f}% (v1: {v1_test_doc['cond_accuracy']*100:.2f}%) | "
        f"macro_f1={test_summary['cond_macro_f1']:.4f} (v1: {v1_test_doc['cond_macro_f1']:.4f})",
        flush=True,
    )
    print(
        f"      Test v2 E2E   : acc={test_summary['end_to_end_accuracy']*100:.2f}% (v1: {v1_test_doc['end_to_end_accuracy']*100:.2f}%)",
        flush=True,
    )
    print(
        f"      Test v2 Mir   : cond_acc={test_summary['test_mirrored_bilateral_check']['cond_accuracy']*100:.2f}% | "
        f"macro_f1={test_summary['test_mirrored_bilateral_check']['cond_macro_f1']:.4f} | reload_bitwise=True",
        flush=True,
    )

    metadata_doc = {
        "experiment": "direction_a_phase2b_landmark_geom_v2_svm",
        "candidate_id": winner_id,
        "model_path": str(model_path.relative_to(PROJECT_ROOT).as_posix()),
        "model_sha256": model_sha256,
        "model_family": bundle["model_family"],
        "pipeline_steps": ["scaler:StandardScaler", f"clf:{type(winner_pipe.named_steps['clf']).__name__}"],
        "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
        "base_xy_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
        "feature_breakdown": {
            "xy_normalized_coords": 42,
            "pairwise_fingertip_distances": 10,
            "fingertip_to_wrist_distances": 5,
            "fingertip_to_mcp_distances": 5,
            "thumb_tip_to_knuckle_distances": 5,
            "joint_bend_cosines": 15,
            "adjacent_finger_splay_cosines": 4,
            "total": 86,
        },
        "mediapipe_version": cache["mediapipe_version"],
        "border_pad_ratio": float(cache["border_pad_ratio"]),
        "min_detection_confidence": float(cache["min_detection_confidence"]),
        "hand_landmarker_model": cache["hand_landmarker_model"],
        "class_names": class_names,
        "class_count": len(class_names),
        "aug_mode": winner["aug_mode"],
        "rotation_angles_deg": winner["rotation_angles_deg"],
        "augment_horizontal_flip": True,
        "hyperparameters": hyperparams,
        "split_config": split_config,
        "fit_seconds": float(winner["fit_seconds"]),
        "convergence_warnings": int(winner["convergence_warnings"]),
        "validation_selection_summary": {
            "v1_baseline_combined_macro_f1": v1_val_comb_f1,
            "val_orig_cond_accuracy": winner["val_orig_cond_accuracy"],
            "val_orig_cond_macro_f1": winner["val_orig_cond_macro_f1"],
            "val_orig_e2e_accuracy": winner["val_orig_e2e_accuracy"],
            "val_mirrored_cond_accuracy": winner["val_mirrored_cond_accuracy"],
            "val_mirrored_cond_macro_f1": winner["val_mirrored_cond_macro_f1"],
            "val_mirrored_e2e_accuracy": winner["val_mirrored_e2e_accuracy"],
            "combined_val_cond_macro_f1": winner["combined_val_cond_macro_f1"],
            "delta_vs_v1_combined_f1": winner["delta_vs_v1_combined_f1"],
        },
        "created_at": bundle["created_at"],
    }
    metadata_path.write_text(json.dumps(metadata_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 4: 3-way webcam benchmark (v2 vs v1 vs HOG Method 3B)
    print("\n[4/4] Benchmarking 3-way (Landmark v2 vs Landmark v1 vs HOG 3B) on real webcam sets...", flush=True)
    v1_bundle = joblib.load(args.v1_model)
    v1_pipe: Pipeline = v1_bundle["model"]
    webcam_report, webcam_csv_rows = run_3way_webcam_benchmark(
        v2_pipe=reloaded_v2_pipe,
        v1_pipe=v1_pipe,
        class_names=class_names,
        datasets=datasets,
        excluded_info=excluded_info,
        webcam_cache=webcam_cache,
        border_pad_ratio=float(cache["border_pad_ratio"]),
        min_detection_confidence=float(cache["min_detection_confidence"]),
    )
    for sid, sd in webcam_report["sets"].items():  # type: ignore[union-attr]
        print(
            f"      {sid:32s} | det={sd['detected_samples']:2d}/{sd['total_samples']:2d} | "
            f"v2 e2e={sd['v2_correct']:2d}/{sd['total_samples']:2d} ({sd['v2_e2e_pct']:5.2f}%, cond={sd['v2_cond_pct']:5.2f}%) | "
            f"v1 e2e={sd['v1_correct']:2d}/{sd['total_samples']:2d} ({sd['v1_e2e_pct']:5.2f}%) | "
            f"HOG3B e2e={sd['hog_3b_correct']:2d}/{sd['total_samples']:2d} ({sd['hog_3b_e2e_pct']:5.2f}%)",
            flush=True,
        )

    # Save Phase 2b reports strictly inside results/svm/experiments/landmark_asl_v2/
    diag_path = args.report_dir / "phase2b_cd_geometry_diagnosis.json"
    val_json_path = args.report_dir / "phase2b_validation_comparison.json"
    val_csv_path = args.report_dir / "phase2b_validation_comparison.csv"
    test_json_path = args.report_dir / "phase2b_test_evaluation.json"
    test_csv_path = args.report_dir / "phase2b_test_per_class.csv"
    wb_json_path = args.report_dir / "phase2b_webcam_3way_benchmark.json"
    wb_csv_path = args.report_dir / "phase2b_webcam_3way_samples.csv"

    diag_path.write_text(json.dumps(cd_diag, indent=2, ensure_ascii=False), encoding="utf-8")
    val_json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "v1_baseline_combined_macro_f1": v1_val_comb_f1,
                "validation_prefix_groups": ["P8", "P10"],
                "validation_total": val_total,
                "validation_detected": int(splits["validation"]["detected"]),
                "selected_candidate_id": winner_id,
                "selected_candidate": winner,
                "candidates_ranked": ranked,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    val_csv_fields = [
        "rank",
        "candidate_id",
        "model_family",
        "kernel",
        "C",
        "gamma",
        "aug_mode",
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
        "delta_vs_v1_combined_f1",
        "combined_watch_recall",
    ]
    with val_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_csv_fields)
        w.writeheader()
        for rank_idx, r in enumerate(ranked, start=1):
            row_out = {k: r[k] for k in val_csv_fields if k != "rank"}
            row_out["rank"] = rank_idx
            w.writerow(row_out)

    test_json_path.write_text(json.dumps(test_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with test_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test_per_class_rows[0].keys()))
        w.writeheader()
        w.writerows(test_per_class_rows)

    wb_json_path.write_text(json.dumps(webcam_report, indent=2, ensure_ascii=False), encoding="utf-8")
    with wb_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(webcam_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(webcam_csv_rows)

    print("\nSaved Phase 2b artifacts:", flush=True)
    for p in (model_path, metadata_path, diag_path, val_json_path, val_csv_path, test_json_path, test_csv_path, wb_json_path, wb_csv_path):
        print(f"  - {p.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    print("=============================================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
