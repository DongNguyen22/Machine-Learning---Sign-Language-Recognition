"""SEPARATE EXPERIMENT: MediaPipe 21 Hand Landmarks (42-dim) + SVM (LinearSVC / RBF SVC).

Inspired by the landmark feature step of `computervisioneng/sign-language-detector-python`
(`create_dataset.py`), adapted to SVM (`StandardScaler + LinearSVC` / `SVC(kernel='rbf')`)
— NEVER Random Forest and NEVER mixed into the 1764-dim HOG model.

Feature contract (identical in training and `src/model/predict_landmark_svm.py`):
- Input BGR hand image -> canonicalize orientation -> add 50% neutral border padding
  (prevents MediaPipe misses on tight crops) -> MediaPipe Tasks `HandLandmarker`
  (`results/svm/hand_landmarker.task`) -> 21 `(x, y)` landmarks -> wrist-centered and
  L2-span-normalized 42-dim `float32` vector (`mediapipe_21_xy_wrist_norm_v1`).
- Images where MediaPipe fails to detect a hand are counted and reported honestly as
  undetected (never silently dropped from test denominator when reporting end-to-end rate!).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import warnings

import cv2
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_webcam_svm import (
    compute_detailed_metrics,
    load_and_verify_split_manifest,
    sha256_file,
)
from src.model.train_svm import RANDOM_STATE
from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    create_hand_landmarker,
    extract_mediapipe_landmarks_feature_from_bgr,
)


def extract_landmark_split(
    records: list[dict[str, object]],
    class_names: list[str],
    detector,
    border_pad_ratio: float = 0.5,
) -> dict[str, object]:
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    X_detected: list[np.ndarray] = []
    y_detected: list[int] = []
    detected_flags: list[bool] = []
    undetected_folders: list[str] = []

    for rec in records:
        bgr = cv2.imread(str(rec["crop_path"]))
        if bgr is None:
            raise ValueError(f"Cannot read image: {rec['crop_path']}")
        feat, info = extract_mediapipe_landmarks_feature_from_bgr(
            bgr,
            detector=detector,
            metadata=rec["disk_meta"],  # type: ignore[arg-type]
            flip_horizontal=False,
            border_pad_ratio=border_pad_ratio,
        )
        if feat is None:
            detected_flags.append(False)
            undetected_folders.append(str(rec["folder_name"]))
        else:
            detected_flags.append(True)
            X_detected.append(feat)
            y_detected.append(label_to_idx[str(rec["label"])])

    return {
        "X": np.asarray(X_detected, dtype=np.float32) if X_detected else np.empty((0, LANDMARK_FEATURE_DIM), dtype=np.float32),
        "y": np.asarray(y_detected, dtype=np.int64),
        "detected_flags": detected_flags,
        "detected_count": len(X_detected),
        "total_count": len(records),
        "detection_rate": float(len(X_detected) / len(records)) if records else 0.0,
        "undetected_folders": undetected_folders,
    }


def build_landmark_svm(kernel: str, c_val: float, max_iter: int) -> Pipeline:
    if kernel == "linear":
        clf = LinearSVC(C=c_val, random_state=RANDOM_STATE, max_iter=max_iter, dual="auto")
    elif kernel == "rbf":
        clf = SVC(C=c_val, kernel="rbf", gamma="scale", random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported SVM kernel: {kernel}")
    return Pipeline(steps=[("scaler", StandardScaler()), ("classifier", clf)])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--samples-dir", type=Path, required=True)
    p.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_LANDMARKER_MODEL)
    p.add_argument("--border-pad-ratio", type=float, default=0.5)
    p.add_argument("--kernels", nargs="+", choices=["linear", "rbf"], default=["linear", "rbf"])
    p.add_argument("--c-grid", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    p.add_argument("--max-iter", type=int, default=10_000)
    p.add_argument("--model-output", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=Path("results/svm/landmark_svm_experiments"))
    p.add_argument("--allow-subset-for-tests", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)

    records, class_names = load_and_verify_split_manifest(
        split_csv=args.split_csv,
        samples_dir=args.samples_dir,
        require_all_36_classes=not args.allow_subset_for_tests,
    )
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "validation"]
    test_recs = [r for r in records if r["split"] == "test"]

    detector = create_hand_landmarker(model_path=args.hand_model)
    try:
        tr_data = extract_landmark_split(train_recs, class_names, detector, args.border_pad_ratio)
        va_data = extract_landmark_split(val_recs, class_names, detector, args.border_pad_ratio)
        te_data = extract_landmark_split(test_recs, class_names, detector, args.border_pad_ratio)
    finally:
        detector.close()

    if len(np.unique(tr_data["y"])) < len(class_names):
        missing_after_mp = sorted(set(range(len(class_names))) - set(np.unique(tr_data["y"]).tolist()))
        raise ValueError(
            f"MediaPipe failed to detect hands for all classes in train; missing class indices: {missing_after_mp}"
        )
    if va_data["detected_count"] == 0 or te_data["detected_count"] == 0:
        raise ValueError("MediaPipe detected 0 hands in validation or test split; cannot evaluate.")

    candidates: list[dict[str, object]] = []
    fitted: dict[tuple[str, float], Pipeline] = {}
    va_true_lbl = [class_names[i] for i in va_data["y"]]

    for kernel in args.kernels:
        for c_val in args.c_grid:
            pipe = build_landmark_svm(kernel, float(c_val), args.max_iter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                t0 = time.perf_counter()
                pipe.fit(tr_data["X"], tr_data["y"])
                fit_s = time.perf_counter() - t0
            va_pred = pipe.predict(va_data["X"])
            va_pred_lbl = [class_names[i] for i in va_pred]
            m = compute_detailed_metrics(va_true_lbl, va_pred_lbl, class_names)
            candidates.append(
                {
                    "kernel": kernel,
                    "c": float(c_val),
                    "fit_seconds": round(fit_s, 4),
                    "convergence_warnings": len(caught),
                    "validation_accuracy_on_detected": m["accuracy"],
                    "validation_macro_f1_on_detected": m["macro_f1"],
                }
            )
            fitted[(kernel, float(c_val))] = pipe

    candidates.sort(
        key=lambda r: (
            -float(r["validation_macro_f1_on_detected"]),
            -float(r["validation_accuracy_on_detected"]),
            float(r["c"]),
        )
    )
    best = candidates[0]
    best_pipe = fitted[(str(best["kernel"]), float(best["c"]))]

    # Evaluate ONCE on independent test (both on detected subset AND end-to-end counting misses as wrong)
    te_pred = best_pipe.predict(te_data["X"])
    te_true_det = [class_names[i] for i in te_data["y"]]
    te_pred_det = [class_names[i] for i in te_pred]
    det_metrics = compute_detailed_metrics(te_true_det, te_pred_det, class_names)

    end_to_end_true = [str(r["label"]) for r in test_recs]
    end_to_end_pred: list[str] = []
    pred_iter = iter(te_pred_det)
    for flag in te_data["detected_flags"]:
        end_to_end_pred.append(next(pred_iter) if flag else "__NO_HAND_DETECTED__")
    end_to_end_correct = sum(1 for t, pr in zip(end_to_end_true, end_to_end_pred) if t == pr)
    end_to_end_accuracy = float(end_to_end_correct / len(end_to_end_true)) if end_to_end_true else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output = args.model_output or Path(f"models/svm/experiments/svm_landmarks_{ts}.joblib")
    model_output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.report_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best_pipe,
        "model_family": f"StandardScaler + {'LinearSVC' if best['kernel'] == 'linear' else 'SVC(rbf)'}",
        "feature_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_count": LANDMARK_FEATURE_DIM,
        "border_pad_ratio": float(args.border_pad_ratio),
        "hand_landmarker_model": str(args.hand_model),
        "class_names": np.asarray(class_names, dtype=str),
        "kernel": best["kernel"],
        "c": float(best["c"]),
        "split_csv": str(args.split_csv),
        "split_csv_sha256": sha256_file(args.split_csv),
        "is_full_36_class_model": (len(class_names) == 36 and not args.allow_subset_for_tests),
    }
    joblib.dump(bundle, model_output)

    report = {
        "experiment": "separate_mediapipe_landmarks_svm",
        "note": "Thí nghiệm tách biệt hoàn toàn với mô hình HOG 1764 chiều; dùng SVM (LinearSVC/SVC), không dùng Random Forest.",
        "feature_schema": LANDMARK_FEATURE_SCHEMA,
        "feature_count": LANDMARK_FEATURE_DIM,
        "detection_rates": {
            "train": {"detected": tr_data["detected_count"], "total": tr_data["total_count"], "rate": tr_data["detection_rate"]},
            "validation": {"detected": va_data["detected_count"], "total": va_data["total_count"], "rate": va_data["detection_rate"]},
            "test": {"detected": te_data["detected_count"], "total": te_data["total_count"], "rate": te_data["detection_rate"], "undetected_folders": te_data["undetected_folders"]},
        },
        "validation_candidates": candidates,
        "selected_on_validation": best,
        "test_metrics_on_detected_hands": det_metrics,
        "test_end_to_end_accuracy_counting_misses_as_wrong": end_to_end_accuracy,
        "saved_model_path": str(model_output.resolve()),
    }
    report_path = run_dir / "landmark_svm_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("================ SEPARATE MEDIAPIPE LANDMARKS + SVM ================")
    print(f"Detection rate (test)  : {te_data['detected_count']}/{te_data['total_count']} ({te_data['detection_rate']*100:.1f}%)")
    print(f"Selected on validation : kernel={best['kernel']}, C={best['c']}")
    print(f"Test Acc (detected)    : {det_metrics['accuracy']*100:.2f}% | Macro F1: {det_metrics['macro_f1']:.4f}")
    print(f"Test Acc (end-to-end)  : {end_to_end_accuracy*100:.2f}% (counting misses as wrong)")
    print(f"Saved separate model   : {model_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
