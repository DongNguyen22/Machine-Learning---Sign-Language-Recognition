"""Hướng A — Đợt 3: Controlled comparison of splitting Letters (A–Z, 26 classes) vs Digits (0–9, 10 classes).

Compares strictly on `validation (P8, P10)` (`val_orig` + `val_mirrored`):
- Baseline: `v2_36_full` (`svm_landmark_asl_v2.joblib`, free 36-class prediction)
- Method 1: `v2_36_masked` (`svm_landmark_asl_v2.joblib`, argmax decision_function masked to A–Z or 0–9)
- Method 2: Dedicated `SVM_AZ` (26 classes) & `SVM_09` (10 classes) using
  `Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf', ...))])` on `mediapipe_21_xy_geom_v2` (86D)
  with horizontal flip augmentation ONLY on `train`.

Also diagnoses the 4 real webcam confusions reported by the user (`I <-> J`, `K -> 2`, `V -> 2`, `X -> 1`),
runs ONE-TIME final evaluation on `test (P2, P7)`, and runs a 4-way benchmark on real confirmed webcam
`A/B/C/D` sets (`HOG 3B` vs `v2_36_full` vs `v2_36_masked (A–Z)` vs `SVM_AZ`).

Safety guarantees:
- Reads Phase 1 cache (`results/svm/experiments/landmark_asl_v1/asl_landmarks_cache.npz`) read-only.
- NEVER modifies `svm_api_server.py`, frontend, 7 legacy models in `models/svm/`, `landmark_asl_v1`,
  `landmark_asl_v2`, or any webcam images/labels/reports.
- Writes ONLY to `models/svm/experiments/landmark_asl_v3_split/` and `results/svm/experiments/landmark_asl_v3_split/`.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import warnings

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
from sklearn.svm import SVC

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
from scripts.train_asl_landmark_svm_phase2b import (
    V2_MODEL_DIR,
    extract_webcam_xy_cache,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.model.train_svm import RANDOM_STATE
from src.preprocess.webcam_features import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    extract_landmark_geometric_features_v2,
    flip_landmark_features_horizontal,
)

V2_MODEL_PATH = V2_MODEL_DIR / "svm_landmark_asl_v2.joblib"
V3_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v3_split"
V3_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v3_split"

DIGITS_09 = [str(i) for i in range(10)]
LETTERS_AZ = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TARGET_4_ERRORS = [("I", "J"), ("J", "I"), ("K", "2"), ("V", "2"), ("X", "1")]


def verify_subset_zero_leakage(
    cache: dict[str, object],
    subset_name: str,
    subset_classes: list[str],
) -> dict[str, object]:
    """Slice cache to `subset_classes` and strictly verify 0 leakage and full class coverage."""
    class_names: list[str] = cache["class_names"]  # type: ignore[assignment]
    y_all: np.ndarray = cache["y_all"]  # type: ignore[assignment]
    detected: np.ndarray = cache["detected"]  # type: ignore[assignment]
    new_split: np.ndarray = cache["new_split"]  # type: ignore[assignment]
    prefix_group: np.ndarray = cache["prefix_group"]  # type: ignore[assignment]
    file_sha256: np.ndarray = cache["file_sha256"]  # type: ignore[assignment]

    global_ids = [class_names.index(c) for c in subset_classes]
    global_to_local = {gid: lid for lid, gid in enumerate(global_ids)}
    in_subset = np.isin(y_all, global_ids)

    split_masks: dict[str, dict[str, object]] = {}
    groups_by_split: dict[str, set[str]] = {}
    shas_by_split: dict[str, set[str]] = {}

    for sp in ("train", "validation", "test"):
        m_all = in_subset & (new_split == sp)
        m_det = m_all & detected
        groups = sorted(
            set(prefix_group[m_all].tolist()),
            key=lambda s: (int(s[1:]) if len(s) > 1 and s[1:].isdigit() else 999, s),
        )
        groups_by_split[sp] = set(groups)
        shas_by_split[sp] = set(file_sha256[m_all].tolist())

        classes_total = set(y_all[m_all].tolist())
        classes_det = set(y_all[m_det].tolist())
        assert len(classes_total) == len(subset_classes), f"{subset_name}/{sp} missing classes in total"
        assert len(classes_det) == len(subset_classes), f"{subset_name}/{sp} missing classes in detected"

        y_det_global = y_all[m_det]
        y_det_local = np.array([global_to_local[int(v)] for v in y_det_global], dtype=np.int32)
        y_all_local = np.array([global_to_local[int(v)] for v in y_all[m_all]], dtype=np.int32)

        split_masks[sp] = {
            "mask_all": m_all,
            "mask_det": m_det,
            "total": int(np.sum(m_all)),
            "detected": int(np.sum(m_det)),
            "undetected": int(np.sum(m_all & (~detected))),
            "detection_rate": round(float(np.sum(m_det) / np.sum(m_all)), 6),
            "prefix_groups": groups,
            "y_det_global": y_det_global,
            "y_det_local": y_det_local,
            "y_all_local": y_all_local,
        }

    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert len(groups_by_split[a] & groups_by_split[b]) == 0, f"{subset_name} group leak {a}-{b}"
        assert len(shas_by_split[a] & shas_by_split[b]) == 0, f"{subset_name} sha256 leak {a}-{b}"

    return {
        "subset_name": subset_name,
        "class_names": subset_classes,
        "global_class_ids": global_ids,
        "global_to_local": global_to_local,
        "splits": split_masks,
    }


def predict_v2_masked_local(
    v2_pipe: Pipeline,
    X_v2: np.ndarray,
    global_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Mask v2 36-class OvR decision_function to `global_ids` and return (local_pred_idx, masked_scores)."""
    scores_36 = np.asarray(v2_pipe.decision_function(X_v2), dtype=np.float64)
    clf_classes = np.asarray(v2_pipe.named_steps["clf"].classes_, dtype=np.int64)
    col_positions = [int(np.where(clf_classes == gid)[0][0]) for gid in global_ids]
    masked_scores = scores_36[:, col_positions]
    local_pred = np.argmax(masked_scores, axis=1).astype(np.int32)
    return local_pred, masked_scores


def compute_subset_metrics(
    y_true_labels: list[str],
    y_pred_labels: list[str],
    subset_classes: list[str],
    total_count: int,
) -> dict[str, object]:
    n_det = len(y_true_labels)
    n_corr = sum(1 for t, p in zip(y_true_labels, y_pred_labels) if t == p)
    cond_acc = float(n_corr / n_det) if n_det else 0.0
    e2e_acc = float(n_corr / total_count) if total_count else 0.0
    cond_macro_f1 = float(
        f1_score(y_true_labels, y_pred_labels, labels=subset_classes, average="macro", zero_division=0)
    )
    out_of_subset = sum(1 for p in y_pred_labels if p not in subset_classes)
    return {
        "detected": n_det,
        "total": int(total_count),
        "correct": n_corr,
        "out_of_subset_predictions": int(out_of_subset),
        "cond_accuracy": round(cond_acc, 6),
        "cond_macro_f1": round(cond_macro_f1, 6),
        "end_to_end_accuracy": round(e2e_acc, 6),
    }


def build_rbf_grid() -> list[dict[str, object]]:
    specs = []
    for c_val in (1.0, 10.0, 30.0):
        for gamma in ("scale", 0.005, 0.01):
            specs.append(
                {
                    "candidate_id": f"rbf_C{c_val:g}_g{gamma}_flip",
                    "C": float(c_val),
                    "gamma": gamma,
                }
            )
    return specs


def run_validation_comparison_for_subset(
    subset: dict[str, object],
    X_all: np.ndarray,
    v2_pipe: Pipeline,
    all_36_class_names: list[str],
) -> tuple[dict[str, object], dict[str, object], Pipeline]:
    """Compare v2_36_full vs v2_36_masked vs 9 dedicated RBF SVC candidates strictly on validation."""
    sub_name = str(subset["subset_name"])
    sub_classes: list[str] = subset["class_names"]  # type: ignore[assignment]
    global_ids: list[int] = subset["global_class_ids"]  # type: ignore[assignment]
    sp_tr = subset["splits"]["train"]  # type: ignore[index]
    sp_va = subset["splits"]["validation"]  # type: ignore[index]

    X_tr_xy = X_all[sp_tr["mask_det"]]
    y_tr_local = sp_tr["y_det_local"]
    X_tr_v2_flip = extract_landmark_geometric_features_v2(
        np.vstack([X_tr_xy, flip_landmark_features_horizontal(X_tr_xy)])
    )
    y_tr_flip = np.concatenate([y_tr_local, y_tr_local]).astype(np.int32)

    X_va_xy = X_all[sp_va["mask_det"]]
    X_va_v2_orig = extract_landmark_geometric_features_v2(X_va_xy)
    X_va_v2_mir = extract_landmark_geometric_features_v2(flip_landmark_features_horizontal(X_va_xy))
    y_va_lbl = [sub_classes[int(i)] for i in sp_va["y_det_local"]]
    va_total = int(sp_va["total"])

    rows: list[dict[str, object]] = []

    # 1. Baseline: v2_36_full (free 36-class predictions)
    pred_full_orig = [all_36_class_names[int(i)] for i in v2_pipe.predict(X_va_v2_orig)]
    pred_full_mir = [all_36_class_names[int(i)] for i in v2_pipe.predict(X_va_v2_mir)]
    m_fo = compute_subset_metrics(y_va_lbl, pred_full_orig, sub_classes, va_total)
    m_fm = compute_subset_metrics(y_va_lbl, pred_full_mir, sub_classes, va_total)
    rows.append(
        {
            "method_type": "baseline_v2_36_full",
            "candidate_id": f"{sub_name}__v2_36_full",
            "C": 1.0,
            "gamma": "0.005",
            "fit_seconds": 0.0,
            "convergence_warnings": 0,
            "val_orig_cond_accuracy": m_fo["cond_accuracy"],
            "val_orig_cond_macro_f1": m_fo["cond_macro_f1"],
            "val_orig_e2e_accuracy": m_fo["end_to_end_accuracy"],
            "val_orig_out_of_subset": m_fo["out_of_subset_predictions"],
            "val_mir_cond_accuracy": m_fm["cond_accuracy"],
            "val_mir_cond_macro_f1": m_fm["cond_macro_f1"],
            "val_mir_e2e_accuracy": m_fm["end_to_end_accuracy"],
            "combined_val_cond_macro_f1": round(0.5 * (float(m_fo["cond_macro_f1"]) + float(m_fm["cond_macro_f1"])), 6),
            "min_val_cond_macro_f1": round(min(float(m_fo["cond_macro_f1"]), float(m_fm["cond_macro_f1"])), 6),
            "combined_val_cond_accuracy": round(0.5 * (float(m_fo["cond_accuracy"]) + float(m_fm["cond_accuracy"])), 6),
        }
    )

    # 2. Method 1: v2_36_masked
    m_orig_idx, _ = predict_v2_masked_local(v2_pipe, X_va_v2_orig, global_ids)
    m_mir_idx, _ = predict_v2_masked_local(v2_pipe, X_va_v2_mir, global_ids)
    pred_mask_orig = [sub_classes[int(i)] for i in m_orig_idx]
    pred_mask_mir = [sub_classes[int(i)] for i in m_mir_idx]
    m_mo = compute_subset_metrics(y_va_lbl, pred_mask_orig, sub_classes, va_total)
    m_mm = compute_subset_metrics(y_va_lbl, pred_mask_mir, sub_classes, va_total)
    rows.append(
        {
            "method_type": "method1_v2_36_masked",
            "candidate_id": f"{sub_name}__v2_36_masked",
            "C": 1.0,
            "gamma": "0.005",
            "fit_seconds": 0.0,
            "convergence_warnings": 0,
            "val_orig_cond_accuracy": m_mo["cond_accuracy"],
            "val_orig_cond_macro_f1": m_mo["cond_macro_f1"],
            "val_orig_e2e_accuracy": m_mo["end_to_end_accuracy"],
            "val_orig_out_of_subset": 0,
            "val_mir_cond_accuracy": m_mm["cond_accuracy"],
            "val_mir_cond_macro_f1": m_mm["cond_macro_f1"],
            "val_mir_e2e_accuracy": m_mm["end_to_end_accuracy"],
            "combined_val_cond_macro_f1": round(0.5 * (float(m_mo["cond_macro_f1"]) + float(m_mm["cond_macro_f1"])), 6),
            "min_val_cond_macro_f1": round(min(float(m_mo["cond_macro_f1"]), float(m_mm["cond_macro_f1"])), 6),
            "combined_val_cond_accuracy": round(0.5 * (float(m_mo["cond_accuracy"]) + float(m_mm["cond_accuracy"])), 6),
        }
    )

    # 3. Method 2: Dedicated SVM grid
    dedicated_rows: list[dict[str, object]] = []
    fitted_dedicated: dict[str, Pipeline] = {}

    for spec in build_rbf_grid():
        cid = f"{sub_name}__{spec['candidate_id']}"
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=float(spec["C"]),
                        kernel="rbf",
                        gamma=spec["gamma"],  # type: ignore[arg-type]
                        decision_function_shape="ovr",
                        probability=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            t0 = time.perf_counter()
            pipe.fit(X_tr_v2_flip, y_tr_flip)
            fit_s = time.perf_counter() - t0

        po = [sub_classes[int(i)] for i in pipe.predict(X_va_v2_orig)]
        pm = [sub_classes[int(i)] for i in pipe.predict(X_va_v2_mir)]
        mo = compute_subset_metrics(y_va_lbl, po, sub_classes, va_total)
        mm = compute_subset_metrics(y_va_lbl, pm, sub_classes, va_total)
        row = {
            "method_type": f"method2_dedicated_{sub_name}",
            "candidate_id": cid,
            "C": float(spec["C"]),
            "gamma": str(spec["gamma"]),
            "fit_seconds": round(fit_s, 3),
            "convergence_warnings": len(caught),
            "val_orig_cond_accuracy": mo["cond_accuracy"],
            "val_orig_cond_macro_f1": mo["cond_macro_f1"],
            "val_orig_e2e_accuracy": mo["end_to_end_accuracy"],
            "val_orig_out_of_subset": 0,
            "val_mir_cond_accuracy": mm["cond_accuracy"],
            "val_mir_cond_macro_f1": mm["cond_macro_f1"],
            "val_mir_e2e_accuracy": mm["end_to_end_accuracy"],
            "combined_val_cond_macro_f1": round(0.5 * (float(mo["cond_macro_f1"]) + float(mm["cond_macro_f1"])), 6),
            "min_val_cond_macro_f1": round(min(float(mo["cond_macro_f1"]), float(mm["cond_macro_f1"])), 6),
            "combined_val_cond_accuracy": round(0.5 * (float(mo["cond_accuracy"]) + float(mm["cond_accuracy"])), 6),
        }
        dedicated_rows.append(row)
        rows.append(row)
        fitted_dedicated[cid] = pipe
        print(
            f"    {cid:30s} | fit={fit_s:5.2f}s | orig: acc={row['val_orig_cond_accuracy']*100:5.2f}% "
            f"f1={row['val_orig_cond_macro_f1']:.4f} e2e={row['val_orig_e2e_accuracy']*100:5.2f}% | "
            f"mir_f1={row['val_mir_cond_macro_f1']:.4f} | comb_f1={row['combined_val_cond_macro_f1']:.4f}",
            flush=True,
        )

    rank_key = lambda r: (
        -float(r["combined_val_cond_macro_f1"]),
        -float(r["min_val_cond_macro_f1"]),
        -float(r["val_orig_cond_macro_f1"]),
        -float(r["combined_val_cond_accuracy"]),
        float(r["C"]),
        float(r["fit_seconds"]),
    )
    dedicated_ranked = sorted(dedicated_rows, key=rank_key)
    all_ranked = sorted(rows, key=rank_key)

    best_dedicated = dedicated_ranked[0]
    best_dedicated_pipe = fitted_dedicated[str(best_dedicated["candidate_id"])]
    overall_winner = all_ranked[0]

    summary = {
        "subset_name": sub_name,
        "class_count": len(sub_classes),
        "train_detected_raw": int(sp_tr["detected"]),
        "train_samples_fitted": int(len(y_tr_flip)),
        "validation_detected": int(sp_va["detected"]),
        "validation_total": va_total,
        "baseline_v2_36_full": rows[0],
        "method1_v2_36_masked": rows[1],
        "best_method2_dedicated": best_dedicated,
        "overall_validation_winner": overall_winner,
        "all_candidates_ranked": all_ranked,
    }
    return summary, best_dedicated, best_dedicated_pipe


def evaluate_subset_on_test_once(
    subset: dict[str, object],
    X_all: np.ndarray,
    v2_pipe: Pipeline,
    dedicated_pipe: Pipeline,
    all_36_class_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    sub_name = str(subset["subset_name"])
    sub_classes: list[str] = subset["class_names"]  # type: ignore[assignment]
    global_ids: list[int] = subset["global_class_ids"]  # type: ignore[assignment]
    sp_te = subset["splits"]["test"]  # type: ignore[index]

    X_te_xy = X_all[sp_te["mask_det"]]
    X_te_v2_orig = extract_landmark_geometric_features_v2(X_te_xy)
    X_te_v2_mir = extract_landmark_geometric_features_v2(flip_landmark_features_horizontal(X_te_xy))
    y_te_local = sp_te["y_det_local"]
    y_te_all_local = sp_te["y_all_local"]
    y_te_lbl = [sub_classes[int(i)] for i in y_te_local]
    te_total = int(sp_te["total"])

    # 1. Baseline v2_36_full
    p_full_orig = [all_36_class_names[int(i)] for i in v2_pipe.predict(X_te_v2_orig)]
    p_full_mir = [all_36_class_names[int(i)] for i in v2_pipe.predict(X_te_v2_mir)]
    m_full_orig = compute_subset_metrics(y_te_lbl, p_full_orig, sub_classes, te_total)
    m_full_mir = compute_subset_metrics(y_te_lbl, p_full_mir, sub_classes, te_total)

    # 2. Method 1 v2_36_masked
    mask_orig_idx, _ = predict_v2_masked_local(v2_pipe, X_te_v2_orig, global_ids)
    mask_mir_idx, _ = predict_v2_masked_local(v2_pipe, X_te_v2_mir, global_ids)
    p_mask_orig = [sub_classes[int(i)] for i in mask_orig_idx]
    p_mask_mir = [sub_classes[int(i)] for i in mask_mir_idx]
    m_mask_orig = compute_subset_metrics(y_te_lbl, p_mask_orig, sub_classes, te_total)
    m_mask_mir = compute_subset_metrics(y_te_lbl, p_mask_mir, sub_classes, te_total)

    # 3. Method 2 Dedicated
    t0 = time.perf_counter()
    ded_orig_idx = np.asarray(dedicated_pipe.predict(X_te_v2_orig), dtype=np.int32)
    ded_scores = np.asarray(dedicated_pipe.decision_function(X_te_v2_orig), dtype=np.float64)
    infer_s = time.perf_counter() - t0
    ded_mir_idx = np.asarray(dedicated_pipe.predict(X_te_v2_mir), dtype=np.int32)
    p_ded_orig = [sub_classes[int(i)] for i in ded_orig_idx]
    p_ded_mir = [sub_classes[int(i)] for i in ded_mir_idx]
    m_ded_orig = compute_subset_metrics(y_te_lbl, p_ded_orig, sub_classes, te_total)
    m_ded_mir = compute_subset_metrics(y_te_lbl, p_ded_mir, sub_classes, te_total)

    def top_confusions(true_l: list[str], pred_l: list[str], k: int = 10) -> list[dict[str, object]]:
        cnt = Counter((t, p) for t, p in zip(true_l, pred_l) if t != p)
        tot_by_true = Counter(true_l)
        out = [
            {
                "true_label": t,
                "predicted_label": p,
                "count": int(c),
                "pct_of_true_detected": round(c / tot_by_true[t] * 100.0, 2),
            }
            for (t, p), c in cnt.items()
        ]
        out.sort(key=lambda r: (-int(r["count"]), str(r["true_label"]), str(r["predicted_label"])))
        return out[:k]

    # Per-class CSV comparing all 3 methods
    _, rec_full, f1_full, _ = precision_recall_fscore_support(y_te_lbl, p_full_orig, labels=sub_classes, zero_division=0)
    _, rec_mask, f1_mask, _ = precision_recall_fscore_support(y_te_lbl, p_mask_orig, labels=sub_classes, zero_division=0)
    prec_ded, rec_ded, f1_ded, sup_ded = precision_recall_fscore_support(y_te_lbl, p_ded_orig, labels=sub_classes, zero_division=0)

    per_class_rows = []
    for lid, cname in enumerate(sub_classes):
        tot_c = int(np.sum(y_te_all_local == lid))
        det_c = int(sup_ded[lid])
        corr_ded = int(np.sum((y_te_local == lid) & (ded_orig_idx == lid)))
        per_class_rows.append(
            {
                "class_name": cname,
                "support_total": tot_c,
                "support_detected": det_c,
                "v2_36_full_cond_recall": round(float(rec_full[lid]), 6),
                "v2_36_full_cond_f1": round(float(f1_full[lid]), 6),
                "v2_36_masked_cond_recall": round(float(rec_mask[lid]), 6),
                "v2_36_masked_cond_f1": round(float(f1_mask[lid]), 6),
                "dedicated_cond_precision": round(float(prec_ded[lid]), 6),
                "dedicated_cond_recall": round(float(rec_ded[lid]), 6),
                "dedicated_cond_f1": round(float(f1_ded[lid]), 6),
                "dedicated_e2e_recall": round(corr_ded / tot_c, 6) if tot_c else 0.0,
            }
        )

    golden_n = min(64, len(ded_orig_idx))
    summary = {
        "subset_name": sub_name,
        "prefix_groups": ["P2", "P7"],
        "total_samples": te_total,
        "detected_samples": int(sp_te["detected"]),
        "undetected_samples": int(sp_te["undetected"]),
        "detection_rate": sp_te["detection_rate"],
        "v2_36_full": {
            "orig": m_full_orig,
            "mirrored": m_full_mir,
            "top_confusions": top_confusions(y_te_lbl, p_full_orig, 10),
        },
        "v2_36_masked": {
            "orig": m_mask_orig,
            "mirrored": m_mask_mir,
            "top_confusions": top_confusions(y_te_lbl, p_mask_orig, 10),
        },
        "dedicated": {
            "orig": m_ded_orig,
            "mirrored": m_ded_mir,
            "inference_ms_per_sample": round(infer_s / len(ded_orig_idx) * 1000.0, 4),
            "top_confusions": top_confusions(y_te_lbl, p_ded_orig, 10),
        },
        "golden_reload_verification": {
            "slice_size": golden_n,
            "y_pred_first_64": ded_orig_idx[:golden_n].tolist(),
            "decision_scores_first_64_sha256": sha256_array(ded_scores[:golden_n]),
            "full_y_pred_sha256": sha256_array(ded_orig_idx),
            "full_decision_scores_sha256": sha256_array(ded_scores),
        },
        "per_class": per_class_rows,
    }
    return summary, per_class_rows, ded_orig_idx, ded_scores


def diagnose_4_webcam_errors(
    cache: dict[str, object],
    subset_az: dict[str, object],
    v2_pipe: Pipeline,
    svm_az_pipe: Pipeline,
) -> dict[str, object]:
    """Quantify I<->J, K->2, V->2, X->1 on val & test + geometric diagnosis of I vs J in asl_processed."""
    X_all: np.ndarray = cache["X_all"]  # type: ignore[assignment]
    y_all: np.ndarray = cache["y_all"]  # type: ignore[assignment]
    class_names_36: list[str] = cache["class_names"]  # type: ignore[assignment]
    prefix_group: np.ndarray = cache["prefix_group"]  # type: ignore[assignment]
    detected: np.ndarray = cache["detected"]  # type: ignore[assignment]
    az_classes: list[str] = subset_az["class_names"]  # type: ignore[assignment]
    az_global_ids: list[int] = subset_az["global_class_ids"]  # type: ignore[assignment]

    pair_stats_by_split: dict[str, object] = {}
    for sp in ("validation", "test"):
        sp_data = subset_az["splits"][sp]  # type: ignore[index]
        X_v2 = extract_landmark_geometric_features_v2(X_all[sp_data["mask_det"]])
        y_true = [az_classes[int(i)] for i in sp_data["y_det_local"]]

        p_full = [class_names_36[int(i)] for i in v2_pipe.predict(X_v2)]
        mask_idx, _ = predict_v2_masked_local(v2_pipe, X_v2, az_global_ids)
        p_mask = [az_classes[int(i)] for i in mask_idx]
        p_az = [az_classes[int(i)] for i in svm_az_pipe.predict(X_v2)]

        def count_target_pairs(preds: list[str]) -> dict[str, object]:
            pair_counts = {}
            for t_lbl, p_lbl in TARGET_4_ERRORS:
                c = sum(1 for t, p in zip(y_true, preds) if t == t_lbl and p == p_lbl)
                pair_counts[f"{t_lbl}->{p_lbl}"] = int(c)
            per_letter = {}
            for letter in ("I", "J", "K", "V", "X"):
                idxs = [i for i, t in enumerate(y_true) if t == letter]
                corr = sum(1 for i in idxs if preds[i] == letter)
                dist = Counter(preds[i] for i in idxs)
                per_letter[letter] = {
                    "correct": int(corr),
                    "detected": len(idxs),
                    "recall_pct": round(corr / len(idxs) * 100.0, 2) if idxs else 0.0,
                    "pred_dist": dict(dist.most_common(5)),
                }
            return {"pair_counts": pair_counts, "per_letter": per_letter}

        pair_stats_by_split[sp] = {
            "v2_36_full": count_target_pairs(p_full),
            "v2_36_masked": count_target_pairs(p_mask),
            "svm_az": count_target_pairs(p_az),
        }

    # Geometric comparison of I vs J in asl_processed (all detected across train/val/test)
    def ij_geom_summary(X_xy: np.ndarray) -> dict[str, float]:
        pts = X_xy.reshape(-1, 21, 2)
        # Palm axis (0 -> 9) angle from vertical (0, -1)
        palm = pts[:, 9, :] - pts[:, 0, :]
        palm_deg = np.rad2deg(np.arctan2(palm[:, 0], -palm[:, 1]))
        # Pinky ray (17 -> 20) angle from vertical (0, -1)
        pinky = pts[:, 20, :] - pts[:, 17, :]
        pinky_deg = np.rad2deg(np.arctan2(pinky[:, 0], -pinky[:, 1]))
        # Finger extensions d(tip, 0)
        d_thumb = np.linalg.norm(pts[:, 4, :] - pts[:, 0, :], axis=-1)
        d_index = np.linalg.norm(pts[:, 8, :] - pts[:, 0, :], axis=-1)
        d_middle = np.linalg.norm(pts[:, 12, :] - pts[:, 0, :], axis=-1)
        d_ring = np.linalg.norm(pts[:, 16, :] - pts[:, 0, :], axis=-1)
        d_pinky = np.linalg.norm(pts[:, 20, :] - pts[:, 0, :], axis=-1)
        # Pinky tip x relative to wrist
        pinky_dx = pts[:, 20, 0]
        pinky_dy = pts[:, 20, 1]
        return {
            "n": int(len(X_xy)),
            "palm_tilt_deg_mean": round(float(np.mean(palm_deg)), 2),
            "palm_tilt_deg_std": round(float(np.std(palm_deg)), 2),
            "abs_palm_tilt_deg_mean": round(float(np.mean(np.abs(palm_deg))), 2),
            "pinky_angle_deg_mean": round(float(np.mean(pinky_deg)), 2),
            "pinky_angle_deg_std": round(float(np.std(pinky_deg)), 2),
            "abs_pinky_angle_deg_mean": round(float(np.mean(np.abs(pinky_deg))), 2),
            "pinky_dx_mean": round(float(np.mean(pinky_dx)), 4),
            "pinky_dy_mean": round(float(np.mean(pinky_dy)), 4),
            "d_pinky20_wrist0_mean": round(float(np.mean(d_pinky)), 4),
            "d_ring16_wrist0_mean": round(float(np.mean(d_ring)), 4),
            "d_thumb4_wrist0_mean": round(float(np.mean(d_thumb)), 4),
            "d_index8_wrist0_mean": round(float(np.mean(d_index)), 4),
            "d_middle12_wrist0_mean": round(float(np.mean(d_middle)), 4),
        }

    idx_i = class_names_36.index("I")
    idx_j = class_names_36.index("J")
    Xi_all = X_all[(y_all == idx_i) & detected]
    Xj_all = X_all[(y_all == idx_j) & detected]
    Xi_geom = extract_landmark_geometric_features_v2(Xi_all)[:, LANDMARK_FEATURE_DIM:]
    Xj_geom = extract_landmark_geometric_features_v2(Xj_all)[:, LANDMARK_FEATURE_DIM:]

    # Distance between I and J centroids in 42D xy (orientation-sensitive) vs 44D rotation-invariant geom
    xy_centroid_dist = float(np.linalg.norm(Xi_all.mean(axis=0) - Xj_all.mean(axis=0)))
    geom_centroid_dist = float(np.linalg.norm(Xi_geom.mean(axis=0) - Xj_geom.mean(axis=0)))
    # Also compute within-class spread to put centroid distance in perspective (Fisher-like ratio)
    xy_within = 0.5 * (
        float(np.mean(np.linalg.norm(Xi_all - Xi_all.mean(axis=0), axis=1)))
        + float(np.mean(np.linalg.norm(Xj_all - Xj_all.mean(axis=0), axis=1)))
    )
    geom_within = 0.5 * (
        float(np.mean(np.linalg.norm(Xi_geom - Xi_geom.mean(axis=0), axis=1)))
        + float(np.mean(np.linalg.norm(Xj_geom - Xj_geom.mean(axis=0), axis=1)))
    )

    by_prefix = {}
    for pg in sorted(set(prefix_group.tolist()), key=lambda s: int(s[1:]) if s[1:].isdigit() else 999):
        mi = (y_all == idx_i) & (prefix_group == pg) & detected
        mj = (y_all == idx_j) & (prefix_group == pg) & detected
        by_prefix[pg] = {
            "I_palm_tilt_deg": ij_geom_summary(X_all[mi])["palm_tilt_deg_mean"],
            "J_palm_tilt_deg": ij_geom_summary(X_all[mj])["palm_tilt_deg_mean"],
            "I_pinky_angle_deg": ij_geom_summary(X_all[mi])["pinky_angle_deg_mean"],
            "J_pinky_angle_deg": ij_geom_summary(X_all[mj])["pinky_angle_deg_mean"],
        }

    return {
        "four_errors_by_split": pair_stats_by_split,
        "i_vs_j_asl_processed_geometry": {
            "I_summary": ij_geom_summary(Xi_all),
            "J_summary": ij_geom_summary(Xj_all),
            "centroid_separation": {
                "xy_42d_centroid_dist": round(xy_centroid_dist, 4),
                "xy_42d_mean_within_radius": round(xy_within, 4),
                "xy_42d_separation_ratio": round(xy_centroid_dist / max(xy_within, 1e-6), 3),
                "geom_44d_centroid_dist": round(geom_centroid_dist, 4),
                "geom_44d_mean_within_radius": round(geom_within, 4),
                "geom_44d_separation_ratio": round(geom_centroid_dist / max(geom_within, 1e-6), 3),
            },
            "by_prefix_group_tilt_deg": by_prefix,
        },
    }


def run_4way_webcam_benchmark(
    v2_pipe: Pipeline,
    svm_az_pipe: Pipeline,
    all_36_class_names: list[str],
    az_classes: list[str],
    az_global_ids: list[int],
    datasets: list[dict[str, object]],
    excluded_info: dict[str, object],
    webcam_cache: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """4-way webcam A/B/C/D benchmark: HOG 3B vs v2_36_full vs v2_36_masked(A-Z) vs SVM_AZ."""
    hog_predictor = SVMSignPredictor(model_path=PROJECT_ROOT / DEFAULT_MODEL_PATH)
    per_folder: dict[str, dict[str, object]] = {}

    for key, rec in webcam_cache.items():
        true_lbl = str(rec["true_label"])
        meta = rec["meta"]
        hand_det = bool(rec["hand_detected"])

        if hand_det:
            feat_v2 = rec["feat_v2_86"]
            X = feat_v2.reshape(1, -1)
            full_res = predict_landmark_top_k(v2_pipe, feat_v2, all_36_class_names, top_k=3)
            full_pred = str(full_res["predicted_label"])

            mask_idx, _ = predict_v2_masked_local(v2_pipe, X, az_global_ids)
            mask_pred = az_classes[int(mask_idx[0])]

            az_res = predict_landmark_top_k(svm_az_pipe, feat_v2, az_classes, top_k=3)
            az_pred = str(az_res["predicted_label"])
            az_top3 = az_res["top_k"]
        else:
            full_pred = "__NO_HAND__"
            mask_pred = "__NO_HAND__"
            az_pred = "__NO_HAND__"
            az_top3 = []

        hog_res = hog_predictor.predict_image_bgr(
            rec["canonical_bgr"],
            top_k=3,
            include_diagnostics=False,
            experimental_webcam_mode=True,
            already_flipped=False,
        )
        hog_pred = str(hog_res["predicted_label"])

        per_folder[key] = {
            "folder_name": rec["folder_name"],
            "true_label": true_lbl,
            "gesture_instance_id": meta.get("gesture_instance_id") or "",
            "hand_detected": hand_det,
            "hog_3b_pred": hog_pred,
            "hog_3b_correct": (hog_pred == true_lbl),
            "v2_36_full_pred": full_pred,
            "v2_36_full_correct": (hand_det and full_pred == true_lbl),
            "v2_36_masked_pred": mask_pred,
            "v2_36_masked_correct": (hand_det and mask_pred == true_lbl),
            "svm_az_pred": az_pred,
            "svm_az_correct": (hand_det and az_pred == true_lbl),
            "svm_az_top3": az_top3,
        }

    sets_report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []

    for ds in datasets:
        sid = str(ds["set_id"])
        srows = []
        for idx, (sdir, _m, _l) in enumerate(ds["items"], start=1):  # type: ignore[union-attr]
            r = dict(per_folder[str(sdir.resolve())])
            r["stt"] = idx
            r["set_id"] = sid
            srows.append(r)
            csv_rows.append(
                {
                    "set_id": sid,
                    "stt": idx,
                    "folder_name": r["folder_name"],
                    "true_label": r["true_label"],
                    "gesture_instance_id": r["gesture_instance_id"],
                    "hand_detected": r["hand_detected"],
                    "hog_3b_pred": r["hog_3b_pred"],
                    "hog_3b_correct": r["hog_3b_correct"],
                    "v2_36_full_pred": r["v2_36_full_pred"],
                    "v2_36_full_correct": r["v2_36_full_correct"],
                    "v2_36_masked_pred": r["v2_36_masked_pred"],
                    "v2_36_masked_correct": r["v2_36_masked_correct"],
                    "svm_az_pred": r["svm_az_pred"],
                    "svm_az_correct": r["svm_az_correct"],
                    "svm_az_top3": "; ".join(f"{t['label']}:{t['decision_score']}" for t in r["svm_az_top3"]),
                }
            )

        n_tot = len(srows)
        n_det = sum(1 for r in srows if r["hand_detected"])
        methods = ["hog_3b", "v2_36_full", "v2_36_masked", "svm_az"]
        totals = {}
        for m in methods:
            ok = sum(1 for r in srows if r[f"{m}_correct"])
            ok_det = sum(1 for r in srows if r["hand_detected"] and r[f"{m}_correct"])
            totals[m] = {
                "correct": ok,
                "e2e_pct": round(ok / n_tot * 100.0, 2) if n_tot else 0.0,
                "cond_pct": round(ok_det / n_det * 100.0, 2) if n_det else 0.0,
            }

        per_class = {}
        for cls in sorted({str(r["true_label"]) for r in srows}):
            cr = [r for r in srows if r["true_label"] == cls]
            cd = [r for r in cr if r["hand_detected"]]
            entry: dict[str, object] = {"total": len(cr), "detected": len(cd)}
            for m in methods:
                ok = sum(1 for r in cr if r[f"{m}_correct"])
                ok_d = sum(1 for r in cd if r[f"{m}_correct"])
                entry[m] = {
                    "correct": ok,
                    "e2e_pct": round(ok / len(cr) * 100.0, 2),
                    "cond_pct": round(ok_d / len(cd) * 100.0, 2) if cd else 0.0,
                    "pred_dist": dict(Counter(str(r[f"{m}_pred"]) for r in cr)),
                }
            per_class[cls] = entry

        sets_report[sid] = {
            "title": ds["title"],
            "total_samples": n_tot,
            "detected_samples": n_det,
            "detection_rate_pct": round(n_det / n_tot * 100.0, 2) if n_tot else 0.0,
            "summary": totals,
            "per_class": per_class,
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "limitation_note": (
            "Các tập ảnh webcam đã xác nhận hiện có chỉ gồm 4 lớp chữ cái (A, B, C, D); "
            "chưa có tập kiểm thử webcam đã xác nhận cho 0–9 và 22 chữ cái còn lại."
        ),
        "excluded_unconfirmed_sets": excluded_info,
        "sets": sets_report,
    }
    return report, csv_rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================ HƯỚNG A — ĐỢT 3: SPLIT A–Z (26) vs 0–9 (10) ================", flush=True)
    cache = load_and_verify_cache(DEFAULT_CACHE_PATH)
    class_names_36: list[str] = cache["class_names"]  # type: ignore[assignment]
    X_all: np.ndarray = cache["X_all"]  # type: ignore[assignment]

    subset_az = verify_subset_zero_leakage(cache, "AZ_26", LETTERS_AZ)
    subset_09 = verify_subset_zero_leakage(cache, "09_10", DIGITS_09)
    print(
        f"[1/4] Verified leak-free splits: "
        f"A–Z (tr={subset_az['splits']['train']['detected']}/{subset_az['splits']['train']['total']}, "
        f"va={subset_az['splits']['validation']['detected']}/{subset_az['splits']['validation']['total']}, "
        f"te={subset_az['splits']['test']['detected']}/{subset_az['splits']['test']['total']}) | "
        f"0–9 (tr={subset_09['splits']['train']['detected']}/{subset_09['splits']['train']['total']}, "
        f"va={subset_09['splits']['validation']['detected']}/{subset_09['splits']['validation']['total']}, "
        f"te={subset_09['splits']['test']['detected']}/{subset_09['splits']['test']['total']})",
        flush=True,
    )

    v2_bundle = joblib.load(V2_MODEL_PATH)
    v2_pipe: Pipeline = v2_bundle["model"]

    print("\n[2/4] Comparing v2_36_full vs v2_36_masked vs dedicated RBF SVC on validation (P8, P10)...", flush=True)
    print("  --- Subset A–Z (26 classes) ---", flush=True)
    val_az, best_ded_az, svm_az_pipe = run_validation_comparison_for_subset(subset_az, X_all, v2_pipe, class_names_36)
    print(
        f"  => A–Z validation winner: {val_az['overall_validation_winner']['candidate_id']} "
        f"(comb_f1={val_az['overall_validation_winner']['combined_val_cond_macro_f1']:.4f} vs "
        f"full={val_az['baseline_v2_36_full']['combined_val_cond_macro_f1']:.4f}, "
        f"masked={val_az['method1_v2_36_masked']['combined_val_cond_macro_f1']:.4f})",
        flush=True,
    )

    print("  --- Subset 0–9 (10 classes) ---", flush=True)
    val_09, best_ded_09, svm_09_pipe = run_validation_comparison_for_subset(subset_09, X_all, v2_pipe, class_names_36)
    print(
        f"  => 0–9 validation winner: {val_09['overall_validation_winner']['candidate_id']} "
        f"(comb_f1={val_09['overall_validation_winner']['combined_val_cond_macro_f1']:.4f} vs "
        f"full={val_09['baseline_v2_36_full']['combined_val_cond_macro_f1']:.4f}, "
        f"masked={val_09['method1_v2_36_masked']['combined_val_cond_macro_f1']:.4f})",
        flush=True,
    )

    # Save dedicated SVM_AZ and SVM_09 models + metadata into landmark_asl_v3_split/
    V3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    V3_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def save_dedicated_model(
        filename: str,
        subset: dict[str, object],
        best_row: dict[str, object],
        pipe: Pipeline,
        val_summary: dict[str, object],
    ) -> tuple[Path, Path, str]:
        m_path = V3_MODEL_DIR / f"{filename}.joblib"
        meta_path = V3_MODEL_DIR / f"{filename}_metadata.json"
        gamma_val = best_row["gamma"] if best_row["gamma"] == "scale" else float(best_row["gamma"])
        bundle = {
            "model": pipe,
            "candidate_id": best_row["candidate_id"],
            "subset_name": subset["subset_name"],
            "model_family": "Pipeline(StandardScaler + SVC(rbf))",
            "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
            "base_xy_schema": LANDMARK_FEATURE_SCHEMA,
            "feature_count": LANDMARK_GEOM_V2_FEATURE_DIM,
            "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
            "mediapipe_version": cache["mediapipe_version"],
            "border_pad_ratio": float(cache["border_pad_ratio"]),
            "min_detection_confidence": float(cache["min_detection_confidence"]),
            "class_names": np.asarray(subset["class_names"], dtype=str),
            "augment_horizontal_flip": True,
            "hyperparameters": {
                "kernel": "rbf",
                "C": float(best_row["C"]),
                "gamma": gamma_val,
                "decision_function_shape": "ovr",
                "probability": False,
                "random_state": RANDOM_STATE,
            },
            "split_config": {
                "cache_path": str(Path(cache["cache_path"]).relative_to(PROJECT_ROOT).as_posix()),
                "cache_sha256": cache["cache_sha256"],
                "train_prefix_groups": subset["splits"]["train"]["prefix_groups"],
                "validation_prefix_groups": subset["splits"]["validation"]["prefix_groups"],
                "test_prefix_groups": subset["splits"]["test"]["prefix_groups"],
                "train_detected_raw": subset["splits"]["train"]["detected"],
                "train_samples_fitted": int(subset["splits"]["train"]["detected"]) * 2,
            },
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        joblib.dump(bundle, m_path)
        m_sha = sha256_file(m_path)
        meta_doc = {
            "experiment": "direction_a_phase3_split_az_09",
            "subset_name": subset["subset_name"],
            "candidate_id": best_row["candidate_id"],
            "model_path": str(m_path.relative_to(PROJECT_ROOT).as_posix()),
            "model_sha256": m_sha,
            "pipeline_steps": ["scaler:StandardScaler", "clf:SVC"],
            "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
            "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
            "class_names": subset["class_names"],
            "class_count": len(subset["class_names"]),
            "augment_horizontal_flip": True,
            "hyperparameters": bundle["hyperparameters"],
            "split_config": bundle["split_config"],
            "validation_summary": {
                "baseline_v2_36_full": val_summary["baseline_v2_36_full"],
                "method1_v2_36_masked": val_summary["method1_v2_36_masked"],
                "dedicated_best": best_row,
                "overall_validation_winner": val_summary["overall_validation_winner"]["candidate_id"],
            },
            "created_at": bundle["created_at"],
        }
        meta_path.write_text(json.dumps(meta_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        return m_path, meta_path, m_sha

    az_model_path, az_meta_path, az_sha = save_dedicated_model("svm_landmark_az_v3", subset_az, best_ded_az, svm_az_pipe, val_az)
    d09_model_path, d09_meta_path, d09_sha = save_dedicated_model("svm_landmark_09_v3", subset_09, best_ded_09, svm_09_pipe, val_09)

    # Step 3: ONE-TIME final test evaluation + bit-for-bit reload verification + 4-error diagnosis
    print("\n[3/4] Running ONE-TIME final test evaluation (P2, P7) & 4-error diagnosis...", flush=True)
    test_az, az_per_class, az_pred, az_scores = evaluate_subset_on_test_once(
        subset_az, X_all, v2_pipe, svm_az_pipe, class_names_36
    )
    test_09, d09_per_class, d09_pred, d09_scores = evaluate_subset_on_test_once(
        subset_09, X_all, v2_pipe, svm_09_pipe, class_names_36
    )

    # Verify bitwise joblib reload parity on BOTH dedicated models
    reloaded_az: Pipeline = joblib.load(az_model_path)["model"]
    X_te_az_v2 = extract_landmark_geometric_features_v2(X_all[subset_az["splits"]["test"]["mask_det"]])
    assert np.array_equal(az_pred, np.asarray(reloaded_az.predict(X_te_az_v2), dtype=np.int32))
    assert np.array_equal(az_scores, np.asarray(reloaded_az.decision_function(X_te_az_v2), dtype=np.float64))

    reloaded_09: Pipeline = joblib.load(d09_model_path)["model"]
    X_te_09_v2 = extract_landmark_geometric_features_v2(X_all[subset_09["splits"]["test"]["mask_det"]])
    assert np.array_equal(d09_pred, np.asarray(reloaded_09.predict(X_te_09_v2), dtype=np.int32))
    assert np.array_equal(d09_scores, np.asarray(reloaded_09.decision_function(X_te_09_v2), dtype=np.float64))

    test_az["joblib_reload_bitwise_parity"] = {"predict_exact": True, "decision_function_exact": True, "model_sha256": az_sha}
    test_09["joblib_reload_bitwise_parity"] = {"predict_exact": True, "decision_function_exact": True, "model_sha256": d09_sha}

    for label_str, tdoc in [("A–Z (26)", test_az), ("0–9 (10)", test_09)]:
        print(
            f"      Test {label_str}: "
            f"full={tdoc['v2_36_full']['orig']['cond_accuracy']*100:.2f}% (f1={tdoc['v2_36_full']['orig']['cond_macro_f1']:.4f}, e2e={tdoc['v2_36_full']['orig']['end_to_end_accuracy']*100:.2f}%) | "
            f"masked={tdoc['v2_36_masked']['orig']['cond_accuracy']*100:.2f}% (f1={tdoc['v2_36_masked']['orig']['cond_macro_f1']:.4f}, e2e={tdoc['v2_36_masked']['orig']['end_to_end_accuracy']*100:.2f}%) | "
            f"dedicated={tdoc['dedicated']['orig']['cond_accuracy']*100:.2f}% (f1={tdoc['dedicated']['orig']['cond_macro_f1']:.4f}, e2e={tdoc['dedicated']['orig']['end_to_end_accuracy']*100:.2f}%)",
            flush=True,
        )

    err_diag = diagnose_4_webcam_errors(cache, subset_az, v2_pipe, reloaded_az)

    # Step 4: 4-way webcam A/B/C/D benchmark
    print("\n[4/4] Running 4-way webcam A/B/C/D benchmark (HOG 3B vs v2_36_full vs v2_36_masked vs SVM_AZ)...", flush=True)
    datasets, excluded_info = collect_webcam_sample_sets()
    webcam_cache = extract_webcam_xy_cache(
        datasets=datasets,
        hand_model_path=Path(cache["hand_landmarker_model"]),
        border_pad_ratio=float(cache["border_pad_ratio"]),
        min_detection_confidence=float(cache["min_detection_confidence"]),
    )
    wb_report, wb_csv_rows = run_4way_webcam_benchmark(
        v2_pipe=v2_pipe,
        svm_az_pipe=reloaded_az,
        all_36_class_names=class_names_36,
        az_classes=LETTERS_AZ,
        az_global_ids=subset_az["global_class_ids"],  # type: ignore[arg-type]
        datasets=datasets,
        excluded_info=excluded_info,
        webcam_cache=webcam_cache,
    )
    for sid, sd in wb_report["sets"].items():  # type: ignore[union-attr]
        sm = sd["summary"]
        print(
            f"      {sid:32s} | HOG3B={sm['hog_3b']['correct']:2d}/{sd['total_samples']:2d} ({sm['hog_3b']['e2e_pct']:5.2f}%) | "
            f"full={sm['v2_36_full']['correct']:2d}/{sd['total_samples']:2d} ({sm['v2_36_full']['e2e_pct']:5.2f}%) | "
            f"masked={sm['v2_36_masked']['correct']:2d}/{sd['total_samples']:2d} ({sm['v2_36_masked']['e2e_pct']:5.2f}%, cond={sm['v2_36_masked']['cond_pct']:5.2f}%) | "
            f"SVM_AZ={sm['svm_az']['correct']:2d}/{sd['total_samples']:2d} ({sm['svm_az']['e2e_pct']:5.2f}%, cond={sm['svm_az']['cond_pct']:5.2f}%)",
            flush=True,
        )

    # Write all Phase 3 reports strictly into results/svm/experiments/landmark_asl_v3_split/
    val_path = V3_REPORT_DIR / "phase3_validation_comparison.json"
    val_csv_path = V3_REPORT_DIR / "phase3_validation_comparison.csv"
    test_path = V3_REPORT_DIR / "phase3_test_evaluation.json"
    az_csv_path = V3_REPORT_DIR / "phase3_test_az_per_class.csv"
    d09_csv_path = V3_REPORT_DIR / "phase3_test_09_per_class.csv"
    diag_path = V3_REPORT_DIR / "phase3_four_errors_diagnosis.json"
    wb_json_path = V3_REPORT_DIR / "phase3_webcam_4way_benchmark.json"
    wb_csv_path = V3_REPORT_DIR / "phase3_webcam_4way_samples.csv"

    val_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "AZ_26": val_az, "09_10": val_09}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    val_csv_rows = []
    for sub_key, vdoc in (("AZ_26", val_az), ("09_10", val_09)):
        for rk, r in enumerate(vdoc["all_candidates_ranked"], start=1):
            row = {"subset": sub_key, "rank": rk, **r}
            val_csv_rows.append(row)
    with val_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(val_csv_rows)

    test_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "AZ_26": test_az, "09_10": test_09}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for csv_p, rows_c in ((az_csv_path, az_per_class), (d09_csv_path, d09_per_class)):
        with csv_p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_c[0].keys()))
            w.writeheader()
            w.writerows(rows_c)

    diag_path.write_text(json.dumps(err_diag, indent=2, ensure_ascii=False), encoding="utf-8")
    wb_json_path.write_text(json.dumps(wb_report, indent=2, ensure_ascii=False), encoding="utf-8")
    with wb_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wb_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(wb_csv_rows)

    print("\nSaved Phase 3 artifacts:", flush=True)
    for p in (az_model_path, az_meta_path, d09_model_path, d09_meta_path, val_path, val_csv_path, test_path, az_csv_path, d09_csv_path, diag_path, wb_json_path, wb_csv_path):
        print(f"  - {p.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
    print("============================================================================", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
