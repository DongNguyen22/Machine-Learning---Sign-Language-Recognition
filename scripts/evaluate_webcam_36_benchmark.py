"""Independent 36-class webcam benchmark: HOG Method 3B vs Landmark SVM (all_36 & scope-specialized).

Strict evaluation rules:
1. REFUSES to run an official evaluation if ANY sample in `--samples-dir` has
   `label_confirmed != True` (or missing `verified_true_label`).
2. Computes & reports honest dataset quality audit: class coverage (/36), distinct participants,
   distinct gesture instances, border-touch warnings, exact SHA-256 duplicates, and near-duplicates
   (gray MAE <= 8.0).
3. Evaluates all 3 methods on the EXACT same canonicalized samples:
   - Method 1: HOG Method 3B (`models/svm/svm_final_c0_01.joblib`)
   - Method 2: Landmark SVM `all_36` (`models/svm/experiments/landmark_asl_v2/svm_landmark_asl_v2.joblib`)
   - Method 3: Landmark SVM Scope-Specialized (`letters_az` for A–Z, `digits_09` for 0–9)
4. Reports BOTH End-to-End Accuracy (undetected hands counted as wrong) AND Conditional Accuracy
   (detected hands only), broken down by All 36, Letters A–Z (26), Digits 0–9 (10), and per-class.
5. Automatically attaches honest limitation notes if `participant_count < 3` or `classes_covered < 36`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict_landmark_v2 import (
    DEFAULT_LANDMARK_09_MODEL_PATH,
    DEFAULT_LANDMARK_AZ_MODEL_PATH,
    DEFAULT_LANDMARK_V2_MODEL_PATH,
    compute_ovr_heuristic_confidence,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    canonicalize_saved_crop_bgr,
    create_hand_landmarker,
    extract_landmark_geometric_features_v2,
    extract_mediapipe_landmarks_feature_from_bgr,
)

DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "webcam_eval_36"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "webcam_eval_36_v1"
ALL_36_LABELS = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]
DIGITS_09_SET = {str(i) for i in range(10)}
LETTERS_AZ_SET = {chr(ord("A") + i) for i in range(26)}
NEAR_DUP_MAE_THRESHOLD = 8.0


def load_and_audit_eval_samples(samples_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load all saved samples, enforce label_confirmed==True gate, and audit metadata/duplicates."""
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Thư mục đánh giá không tồn tại: {samples_dir}")

    folders = sorted(
        p for p in samples_dir.iterdir() if p.is_dir() and (p / "diagnostic_metadata.json").is_file()
    )
    if not folders:
        raise ValueError(f"Không tìm thấy mẫu nào trong thư mục đánh giá: {samples_dir}")

    unconfirmed: list[str] = []
    missing_label: list[str] = []
    records: list[dict[str, Any]] = []

    seen_sha: dict[str, str] = {}
    exact_dups: list[dict[str, Any]] = []
    near_dups: list[dict[str, Any]] = []
    small_grays: list[tuple[str, np.ndarray]] = []

    participants: set[str] = set()
    instances_all: set[str] = set()
    instances_by_class: dict[str, set[str]] = defaultdict(set)
    border_touch_folders: list[str] = []
    counts_by_class: Counter[str] = Counter()

    for idx, folder in enumerate(folders, start=1):
        meta = json.loads((folder / "diagnostic_metadata.json").read_text(encoding="utf-8"))
        crop_path = folder / "received_crop.png"
        gray_path = folder / "preprocessed_gray_64x64.png"
        if not crop_path.is_file():
            raise FileNotFoundError(f"Thiếu file received_crop.png trong {folder}")

        confirmed = bool(meta.get("label_confirmed", False))
        true_label = str(meta.get("verified_true_label") or "").strip().upper()
        if not confirmed:
            unconfirmed.append(folder.name)
        if true_label not in ALL_36_LABELS:
            missing_label.append(folder.name)

        pid = str(meta.get("participant_id") or "").strip()
        sid = str(meta.get("session_id") or "").strip()
        gid = str(meta.get("gesture_instance_id") or "").strip()
        if pid:
            participants.add(pid)
        inst_key = f"{pid}|{sid}|{gid}" if (pid and sid and gid) else ""
        if inst_key:
            instances_all.add(inst_key)
            if true_label in ALL_36_LABELS:
                instances_by_class[true_label].add(inst_key)

        touches = meta.get("touches_frame_border")
        has_border_warn = bool(meta.get("border_touch_warning", False)) or (
            isinstance(touches, list) and len(touches) > 0
        )
        if has_border_warn:
            border_touch_folders.append(folder.name)

        digest = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        if digest in seen_sha:
            exact_dups.append({"first": seen_sha[digest], "second": folder.name, "sha256": digest})
        else:
            seen_sha[digest] = folder.name

        if gray_path.is_file():
            g_img = cv2.imdecode(np.frombuffer(gray_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if g_img is not None:
                small = cv2.resize(g_img, (32, 32)).astype(np.float32)
                for prev_name, prev_small in small_grays:
                    mae = float(np.mean(np.abs(prev_small - small)))
                    if mae <= NEAR_DUP_MAE_THRESHOLD:
                        near_dups.append({"first": prev_name, "second": folder.name, "gray_mae": round(mae, 4)})
                small_grays.append((folder.name, small))

        if true_label in ALL_36_LABELS:
            counts_by_class[true_label] += 1

        records.append(
            {
                "stt": idx,
                "folder_name": folder.name,
                "crop_path": crop_path,
                "meta": meta,
                "true_label": true_label,
                "label_confirmed": confirmed,
                "participant_id": pid,
                "session_id": sid,
                "gesture_instance_id": gid,
                "gesture_instance_key": inst_key,
                "handedness": str(meta.get("handedness") or "").strip(),
                "border_touch_warning": has_border_warn,
                "touches_frame_border": touches if isinstance(touches, list) else [],
                "image_sha256": digest,
            }
        )

    if unconfirmed or missing_label:
        details = []
        if unconfirmed:
            details.append(f"{len(unconfirmed)}/{len(folders)} mẫu chưa xác nhận nhãn (label_confirmed != True)")
        if missing_label:
            details.append(f"{len(missing_label)}/{len(folders)} mẫu thiếu verified_true_label hợp lệ (0–9, A–Z)")
        raise PermissionError(
            "TỪ CHỐI ĐÁNH GIÁ CHÍNH THỨC: "
            + "; ".join(details)
            + ".\nHãy rà soát ảnh bằng:\n"
            + f'  .\\.venv\\Scripts\\python.exe scripts/manage_webcam_samples.py list --diag-dir "{samples_dir.as_posix()}"\n'
            + f"mở {samples_dir.as_posix()}/review_samples.html hoặc review_contact_sheet.png để đối chiếu nhãn thật,\n"
            + "sau đó xác nhận bằng `confirm-all` hoặc `set --updates ... --confirm-all` rồi mới chạy lại đánh giá."
        )

    present_classes = [c for c in ALL_36_LABELS if counts_by_class[c] > 0]
    missing_classes = [c for c in ALL_36_LABELS if counts_by_class[c] == 0]
    limitation_notes: list[str] = []
    if len(participants) < 3:
        limitation_notes.append(
            f"Tập kiểm thử hiện chỉ có {len(participants)} người chụp ({sorted(participants) or ['không rõ']}); "
            "kết quả phản ánh người chụp hiện tại và chưa đại diện cho độ tổng quát nhiều người (>= 3 người)."
        )
    if len(present_classes) < 36:
        limitation_notes.append(
            f"Tập kiểm thử mới phủ {len(present_classes)}/36 lớp (còn thiếu {len(missing_classes)} lớp: "
            f"{', '.join(missing_classes)}); chưa thể kết luận độ chính xác đầy đủ cả 36 lớp."
        )

    audit = {
        "samples_dir": str(samples_dir.as_posix()),
        "total_samples": len(records),
        "confirmed_samples": len(records),
        "classes_covered_count": len(present_classes),
        "classes_total": 36,
        "missing_classes": missing_classes,
        "distinct_participants_count": len(participants),
        "distinct_participants": sorted(participants),
        "distinct_gesture_instances_count": len(instances_all),
        "distinct_gesture_instances_by_class": {c: len(instances_by_class[c]) for c in ALL_36_LABELS},
        "samples_by_class": {c: int(counts_by_class[c]) for c in ALL_36_LABELS},
        "border_touch_warning_count": len(border_touch_folders),
        "border_touch_folders": border_touch_folders,
        "exact_sha256_duplicate_count": len(exact_dups),
        "exact_sha256_duplicates": exact_dups,
        "near_duplicate_mae_le_8_count": len(near_dups),
        "near_duplicates": near_dups,
        "limitation_notes": limitation_notes,
    }
    return records, audit


def predict_landmark_bundle(bundle: dict[str, Any], feat_86: np.ndarray, top_k: int = 3) -> dict[str, Any]:
    pipe = bundle["model"]
    class_names = [str(c) for c in bundle["class_names"]]
    clf_classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
    X = feat_86.reshape(1, -1)

    pred_idx = int(pipe.predict(X)[0])
    scores = np.asarray(pipe.decision_function(X)[0], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    ppos = int(np.where(clf_classes == pred_idx)[0][0])
    if int(order[0]) != ppos:
        order = np.concatenate([[ppos], order[order != ppos]])
    top_pos = order[: min(top_k, len(clf_classes))]
    conf, rel_w, _ = compute_ovr_heuristic_confidence(scores, top_pos, len(clf_classes))

    top_list = [
        {
            "rank": rk,
            "label": class_names[int(clf_classes[int(pos)])],
            "decision_score": round(float(scores[int(pos)]), 4),
            "relative_weight": round(float(rw), 4),
        }
        for rk, (pos, rw) in enumerate(zip(top_pos, rel_w), start=1)
    ]
    return {
        "predicted_label": class_names[pred_idx],
        "decision_score": float(scores[ppos]),
        "confidence": round(float(conf), 4),
        "top_k": top_list,
    }


def summarize_subset(
    rows: list[dict[str, Any]],
    subset_labels: list[str],
    method_key: str,
) -> dict[str, Any]:
    n_tot = len(rows)
    det_rows = [r for r in rows if r["hand_detected"]]
    n_det = len(det_rows)

    e2e_ok = sum(1 for r in rows if r[f"{method_key}_correct"])
    cond_ok = sum(1 for r in det_rows if r[f"{method_key}_correct"])

    present_labels = [c for c in subset_labels if any(r["true_label"] == c for r in rows)]
    y_true_all = [r["true_label"] for r in rows]
    y_pred_all = [r[f"{method_key}_pred"] for r in rows]
    y_true_det = [r["true_label"] for r in det_rows]
    y_pred_det = [r[f"{method_key}_pred"] for r in det_rows]

    f1_e2e = (
        float(f1_score(y_true_all, y_pred_all, labels=present_labels, average="macro", zero_division=0))
        if present_labels
        else 0.0
    )
    present_det = [c for c in subset_labels if any(r["true_label"] == c for r in det_rows)]
    f1_cond = (
        float(f1_score(y_true_det, y_pred_det, labels=present_det, average="macro", zero_division=0))
        if present_det
        else 0.0
    )

    conf_counter = Counter(
        (r["true_label"], r[f"{method_key}_pred"])
        for r in rows
        if not r[f"{method_key}_correct"]
    )
    top_conf = [
        {"true_label": t, "predicted_label": p, "count": int(c)}
        for (t, p), c in conf_counter.most_common(10)
    ]

    return {
        "total_samples": n_tot,
        "detected_samples": n_det,
        "undetected_samples": n_tot - n_det,
        "detection_rate_pct": round(n_det / n_tot * 100.0, 2) if n_tot else 0.0,
        "correct_e2e": e2e_ok,
        "end_to_end_accuracy_pct": round(e2e_ok / n_tot * 100.0, 2) if n_tot else 0.0,
        "macro_f1_e2e": round(f1_e2e, 4),
        "correct_on_detected": cond_ok,
        "conditional_accuracy_pct": round(cond_ok / n_det * 100.0, 2) if n_det else 0.0,
        "macro_f1_conditional": round(f1_cond, 4),
        "top_confusions": top_conf,
    }


def evaluate_webcam_36(
    samples_dir: Path,
    output_dir: Path,
    hog_model_path: Path = PROJECT_ROOT / DEFAULT_MODEL_PATH,
    landmark_all36_path: Path = DEFAULT_LANDMARK_V2_MODEL_PATH,
    landmark_az_path: Path = DEFAULT_LANDMARK_AZ_MODEL_PATH,
    landmark_09_path: Path = DEFAULT_LANDMARK_09_MODEL_PATH,
    hand_model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL,
    border_pad_ratio: float = 0.5,
    min_detection_confidence: float = 0.3,
) -> dict[str, Any]:
    records, audit = load_and_audit_eval_samples(samples_dir)

    hog_predictor = SVMSignPredictor(model_path=hog_model_path)
    bundle_36 = joblib.load(landmark_all36_path)
    bundle_az = joblib.load(landmark_az_path)
    bundle_09 = joblib.load(landmark_09_path)

    detector = create_hand_landmarker(
        model_path=hand_model_path,
        min_detection_confidence=min_detection_confidence,
        num_hands=1,
    )

    sample_rows: list[dict[str, Any]] = []
    try:
        for rec in records:
            crop_bgr = cv2.imread(str(rec["crop_path"]))
            if crop_bgr is None:
                raise ValueError(f"Không đọc được ảnh: {rec['crop_path']}")
            meta = rec["meta"]
            true_label = str(rec["true_label"])
            canonical_bgr = canonicalize_saved_crop_bgr(crop_bgr, metadata=meta)

            # 1. HOG Method 3B
            hog_res = hog_predictor.predict_image_bgr(
                canonical_bgr,
                top_k=3,
                include_diagnostics=False,
                experimental_webcam_mode=True,
                already_flipped=False,
            )
            hog_pred = str(hog_res["predicted_label"])
            hog_score = round(float(hog_res["predicted_decision_score"]), 4)

            # 2. MediaPipe 21 landmarks -> 86-dim v2 features (exact same formula)
            feat_42, mp_info = extract_mediapipe_landmarks_feature_from_bgr(
                crop_bgr,
                detector=detector,
                metadata=meta,
                flip_horizontal=False,
                border_pad_ratio=border_pad_ratio,
            )
            if feat_42 is None:
                hand_det = False
                mp_hand = ""
                lm36_pred = "__NO_HAND__"
                lm36_conf = 0.0
                lm36_top3: list[dict[str, Any]] = []
                lmsc_pred = "__NO_HAND__"
                lmsc_conf = 0.0
                lmsc_scope = "letters_az" if true_label in LETTERS_AZ_SET else "digits_09"
                lmsc_top3: list[dict[str, Any]] = []
            else:
                hand_det = True
                mp_hand = str(mp_info.get("mediapipe_handedness") or "")
                feat_86 = extract_landmark_geometric_features_v2(feat_42)

                res_36 = predict_landmark_bundle(bundle_36, feat_86, top_k=3)
                lm36_pred = str(res_36["predicted_label"])
                lm36_conf = float(res_36["confidence"])
                lm36_top3 = res_36["top_k"]

                if true_label in LETTERS_AZ_SET:
                    lmsc_scope = "letters_az"
                    res_sc = predict_landmark_bundle(bundle_az, feat_86, top_k=3)
                else:
                    lmsc_scope = "digits_09"
                    res_sc = predict_landmark_bundle(bundle_09, feat_86, top_k=3)
                lmsc_pred = str(res_sc["predicted_label"])
                lmsc_conf = float(res_sc["confidence"])
                lmsc_top3 = res_sc["top_k"]

            sample_rows.append(
                {
                    "stt": rec["stt"],
                    "folder_name": rec["folder_name"],
                    "true_label": true_label,
                    "label_group": "letters_az" if true_label in LETTERS_AZ_SET else "digits_09",
                    "participant_id": rec["participant_id"],
                    "session_id": rec["session_id"],
                    "gesture_instance_id": rec["gesture_instance_id"],
                    "handedness": rec["handedness"],
                    "border_touch_warning": rec["border_touch_warning"],
                    "hand_detected": hand_det,
                    "mediapipe_handedness": mp_hand,
                    "hog_3b_pred": hog_pred,
                    "hog_3b_correct": (hog_pred == true_label),
                    "hog_3b_score": hog_score,
                    "landmark_all36_pred": lm36_pred,
                    "landmark_all36_correct": (hand_det and lm36_pred == true_label),
                    "landmark_all36_conf": lm36_conf,
                    "landmark_all36_top3": "; ".join(f"{t['label']}:{t['decision_score']}" for t in lm36_top3),
                    "landmark_scoped_scope": lmsc_scope,
                    "landmark_scoped_pred": lmsc_pred,
                    "landmark_scoped_correct": (hand_det and lmsc_pred == true_label),
                    "landmark_scoped_conf": lmsc_conf,
                    "landmark_scoped_top3": "; ".join(f"{t['label']}:{t['decision_score']}" for t in lmsc_top3),
                }
            )
    finally:
        detector.close()

    rows_az = [r for r in sample_rows if r["label_group"] == "letters_az"]
    rows_09 = [r for r in sample_rows if r["label_group"] == "digits_09"]
    methods = ("hog_3b", "landmark_all36", "landmark_scoped")

    groups_summary = {
        "all_36": {m: summarize_subset(sample_rows, ALL_36_LABELS, m) for m in methods},
        "letters_az": {m: summarize_subset(rows_az, sorted(LETTERS_AZ_SET), m) for m in methods},
        "digits_09": {m: summarize_subset(rows_09, sorted(DIGITS_09_SET), m) for m in methods},
    }

    per_class_rows: list[dict[str, Any]] = []
    for cls in ALL_36_LABELS:
        cr = [r for r in sample_rows if r["true_label"] == cls]
        cd = [r for r in cr if r["hand_detected"]]
        n_c = len(cr)
        n_d = len(cd)
        entry: dict[str, Any] = {
            "class_name": cls,
            "label_group": "letters_az" if cls in LETTERS_AZ_SET else "digits_09",
            "total": n_c,
            "detected": n_d,
            "detection_rate_pct": round(n_d / n_c * 100.0, 2) if n_c else None,
        }
        for m in methods:
            ok_e2e = sum(1 for r in cr if r[f"{m}_correct"])
            ok_det = sum(1 for r in cd if r[f"{m}_correct"])
            entry[f"{m}_correct"] = ok_e2e
            entry[f"{m}_e2e_pct"] = round(ok_e2e / n_c * 100.0, 2) if n_c else None
            entry[f"{m}_cond_pct"] = round(ok_det / n_d * 100.0, 2) if n_d else None
            entry[f"{m}_preds"] = dict(Counter(str(r[f"{m}_pred"]) for r in cr))
        per_class_rows.append(entry)

    error_rows = [
        r
        for r in sample_rows
        if not (r["hog_3b_correct"] and r["landmark_all36_correct"] and r["landmark_scoped_correct"])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "landmark_feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
        "border_pad_ratio": float(border_pad_ratio),
        "min_detection_confidence": float(min_detection_confidence),
        "models": {
            "hog_3b": str(hog_model_path),
            "landmark_all36": str(landmark_all36_path),
            "landmark_letters_az": str(landmark_az_path),
            "landmark_digits_09": str(landmark_09_path),
        },
        "dataset_audit": audit,
        "summary": groups_summary,
        "per_class": per_class_rows,
        "error_samples_count": len(error_rows),
        "error_samples": error_rows,
    }

    json_path = output_dir / "webcam_eval_36_report.json"
    per_class_csv = output_dir / "webcam_eval_36_per_class.csv"
    samples_csv = output_dir / "webcam_eval_36_samples.csv"
    errors_csv = output_dir / "webcam_eval_36_errors.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pc_fields = [
        "class_name",
        "label_group",
        "total",
        "detected",
        "detection_rate_pct",
        "hog_3b_correct",
        "hog_3b_e2e_pct",
        "hog_3b_cond_pct",
        "landmark_all36_correct",
        "landmark_all36_e2e_pct",
        "landmark_all36_cond_pct",
        "landmark_scoped_correct",
        "landmark_scoped_e2e_pct",
        "landmark_scoped_cond_pct",
    ]
    with per_class_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields)
        w.writeheader()
        for row in per_class_rows:
            w.writerow({k: row[k] for k in pc_fields})

    with samples_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        w.writerows(sample_rows)

    with errors_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        w.writerows(error_rows)

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--hog-model", type=Path, default=PROJECT_ROOT / DEFAULT_MODEL_PATH)
    p.add_argument("--landmark-all36-model", type=Path, default=DEFAULT_LANDMARK_V2_MODEL_PATH)
    p.add_argument("--landmark-az-model", type=Path, default=DEFAULT_LANDMARK_AZ_MODEL_PATH)
    p.add_argument("--landmark-09-model", type=Path, default=DEFAULT_LANDMARK_09_MODEL_PATH)
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER_MODEL)
    p.add_argument("--border-pad-ratio", type=float, default=0.5)
    p.add_argument("--min-detection-confidence", type=float, default=0.3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    try:
        report = evaluate_webcam_36(
            samples_dir=args.samples_dir,
            output_dir=args.output_dir,
            hog_model_path=args.hog_model,
            landmark_all36_path=args.landmark_all36_model,
            landmark_az_path=args.landmark_az_model,
            landmark_09_path=args.landmark_09_model,
            hand_model_path=args.hand_model,
            border_pad_ratio=args.border_pad_ratio,
            min_detection_confidence=args.min_detection_confidence,
        )
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 2

    aud = report["dataset_audit"]
    print("================ ĐÁNH GIÁ TẬP KIỂM THỬ WEBCAM 36 LỚP ================")
    print(
        f"Tổng mẫu: {aud['total_samples']} (100% đã xác nhận) | Lớp phủ: {aud['classes_covered_count']}/36 | "
        f"Người chụp: {aud['distinct_participants_count']} {aud['distinct_participants']} | "
        f"Lần giơ độc lập: {aud['distinct_gesture_instances_count']}"
    )
    print(
        f"Chất lượng: Chạm mép={aud['border_touch_warning_count']} | "
        f"Trùng SHA-256={aud['exact_sha256_duplicate_count']} | Gần trùng (MAE<=8)={aud['near_duplicate_mae_le_8_count']}"
    )
    for grp_key, grp_label in (("all_36", "TỔNG 36 LỚP"), ("letters_az", "CHỮ CÁI A–Z"), ("digits_09", "CHỮ SỐ 0–9")):
        g = report["summary"][grp_key]
        h = g["hog_3b"]
        l36 = g["landmark_all36"]
        lsc = g["landmark_scoped"]
        print(
            f"  [{grp_label:11s}] n={h['total_samples']:3d} (det={l36['detected_samples']:3d}) | "
            f"HOG3B E2E={h['end_to_end_accuracy_pct']:5.2f}% (cond={h['conditional_accuracy_pct']:5.2f}%) | "
            f"LM36 E2E={l36['end_to_end_accuracy_pct']:5.2f}% (cond={l36['conditional_accuracy_pct']:5.2f}%) | "
            f"LM_Scope E2E={lsc['end_to_end_accuracy_pct']:5.2f}% (cond={lsc['conditional_accuracy_pct']:5.2f}%)"
        )
    for note in aud["limitation_notes"]:
        print(f"  [GIỚI HẠN] {note}")
    print(f"Đã xuất báo cáo tại: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
