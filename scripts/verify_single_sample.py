"""Verify a single captured webcam sample folder (from SignVision UI or CLI capture tool).

Checks:
1. Required files exist (`received_crop.png`, `preprocessed_gray_64x64.png`,
   `legacy_gray_64x64.png`, `method_3b_gray_64x64.png`, `diagnostic_metadata.json`).
2. All 6 collection metadata fields (`participant_id`, `session_id`, `gesture_instance_id`,
   `handedness`, `background`, `lighting`), `intended_sign` (0-9/A-Z), and `frame_id` are valid.
3. Exact bitwise parity between saved 64x64 grayscale PNGs and backend recomputation via
   `extract_webcam_hog_feature`.
4. Runs `SVMSignPredictor` (current LinearSVC C=0.01) on the crop (legacy & Method 3B) and
   checks MediaPipe `HandLandmarker` 42-dim landmark extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict_svm import SVMSignPredictor
from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    create_hand_landmarker,
    extract_mediapipe_landmarks_feature_from_bgr,
    extract_webcam_hog_feature,
)

REQUIRED_FILES = (
    "received_crop.png",
    "preprocessed_gray_64x64.png",
    "legacy_gray_64x64.png",
    "method_3b_gray_64x64.png",
    "diagnostic_metadata.json",
)
REQUIRED_META = (
    "participant_id",
    "session_id",
    "gesture_instance_id",
    "handedness",
    "background",
    "lighting",
    "intended_sign",
    "frame_id",
)
VALID_LABELS = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def verify_sample_folder(sample_dir: Path, check_landmarks: bool = True) -> dict[str, object]:
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    errors: list[str] = []
    for fname in REQUIRED_FILES:
        if not (sample_dir / fname).is_file():
            errors.append(f"Missing required file: {fname}")

    meta_path = sample_dir / "diagnostic_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    for key in REQUIRED_META:
        val = meta.get(key)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"Missing or empty metadata field: {key}")

    if meta.get("handedness") not in ("left", "right"):
        errors.append(f"Invalid handedness: {meta.get('handedness')!r}")
    if str(meta.get("intended_sign", "")).strip().upper() not in VALID_LABELS:
        errors.append(f"Invalid intended_sign: {meta.get('intended_sign')!r}")

    crop_bgr = cv2.imread(str(sample_dir / "received_crop.png"))
    if crop_bgr is None or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        errors.append("Unreadable or non-BGR received_crop.png")
        return {"passed": False, "sample_dir": str(sample_dir.resolve()), "errors": errors}

    hog_legacy, calc_legacy_g64 = extract_webcam_hog_feature(crop_bgr, preprocess_mode="legacy", metadata=meta)
    hog_m3b, calc_m3b_g64 = extract_webcam_hog_feature(crop_bgr, preprocess_mode="method_3b", metadata=meta)

    active_mode = "method_3b" if bool(meta.get("experimental_webcam_mode", False)) else "legacy"
    if active_mode == "method_3b":
        expected_active_g64 = calc_m3b_g64
    elif bool(meta.get("flip_input_for_model", False)):
        expected_active_g64 = extract_webcam_hog_feature(
            cv2.flip(crop_bgr, 1),
            preprocess_mode="legacy",
            metadata={"crop_orientation": "canonical_unflipped_raw_camera"},
        )[1]
    else:
        expected_active_g64 = calc_legacy_g64

    for disk_name, expected_g64 in (
        ("preprocessed_gray_64x64.png", expected_active_g64),
        ("legacy_gray_64x64.png", calc_legacy_g64),
        ("method_3b_gray_64x64.png", calc_m3b_g64),
    ):
        p = sample_dir / disk_name
        if p.is_file():
            disk_g64 = cv2.imdecode(np.frombuffer(p.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if disk_g64 is None or not np.array_equal(disk_g64, expected_g64):
                errors.append(f"Bitwise mismatch between saved {disk_name} and backend reconstruction")

    predictor = SVMSignPredictor()
    pred_legacy = predictor.predict_from_features(hog_legacy, top_k=3)
    pred_m3b = predictor.predict_from_features(hog_m3b, top_k=3)

    landmark_info: dict[str, object] = {"checked": False}
    if check_landmarks and DEFAULT_HAND_LANDMARKER_MODEL.is_file():
        det = create_hand_landmarker(DEFAULT_HAND_LANDMARKER_MODEL)
        try:
            lm_vec, lm_meta = extract_mediapipe_landmarks_feature_from_bgr(crop_bgr, det, metadata=meta)
            landmark_info = {
                "checked": True,
                "detected": lm_vec is not None,
                "feature_dim": int(lm_vec.shape[0]) if lm_vec is not None else 0,
                **lm_meta,
            }
        finally:
            det.close()

    return {
        "passed": len(errors) == 0,
        "sample_dir": str(sample_dir.resolve()),
        "crop_shape_hwc": list(crop_bgr.shape),
        "label_confirmed": bool(meta.get("label_confirmed", False)),
        "intended_sign": meta.get("intended_sign"),
        "metadata_6_fields": {k: meta.get(k) for k in REQUIRED_META[:6]},
        "hog_1764_parity": len(errors) == 0,
        "current_model_prediction": {
            "legacy": {
                "predicted_label": pred_legacy["predicted_label"],
                "decision_score": round(float(pred_legacy["predicted_decision_score"]), 4),
            },
            "method_3b": {
                "predicted_label": pred_m3b["predicted_label"],
                "decision_score": round(float(pred_m3b["predicted_decision_score"]), 4),
            },
        },
        "mediapipe_landmarks_check": landmark_info,
        "errors": errors,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample_dir", type=Path, help="Path to a captured sample_* folder")
    p.add_argument("--no-landmarks", action="store_true")
    args = p.parse_args()
    res = verify_sample_folder(args.sample_dir, check_landmarks=not args.no_landmarks)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
