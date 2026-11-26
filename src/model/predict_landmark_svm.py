"""Inference helper for the SEPARATE MediaPipe 21 Hand Landmarks + SVM experiment.

Strictly rejects HOG 1764-dim models so landmark features can never be mixed with the HOG model.
Uses the exact same `extract_mediapipe_landmarks_feature_from_bgr` function used during training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.webcam_features import (
    DEFAULT_HAND_LANDMARKER_MODEL,
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    create_hand_landmarker,
    extract_mediapipe_landmarks_feature_from_bgr,
)


class LandmarkSVMPredictor:
    """Loads a separate 42-dim MediaPipe Landmark SVM bundle and predicts on BGR images."""

    def __init__(
        self,
        model_path: Path | str,
        hand_model_path: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict) or bundle.get("feature_schema") != LANDMARK_FEATURE_SCHEMA:
            raise ValueError(
                f"Model '{self.model_path}' is NOT a MediaPipe landmark SVM model "
                f"(expected feature_schema='{LANDMARK_FEATURE_SCHEMA}', got {bundle.get('feature_schema')!r}). "
                "Never feed landmarks into a HOG model."
            )
        if int(bundle.get("feature_count", -1)) != LANDMARK_FEATURE_DIM:
            raise ValueError(
                f"Expected {LANDMARK_FEATURE_DIM} landmark features, got {bundle.get('feature_count')}"
            )
        self.bundle = bundle
        self.pipeline = bundle["model"]
        self.class_names = np.asarray(bundle["class_names"], dtype=str)
        self.border_pad_ratio = float(bundle.get("border_pad_ratio", 0.5))
        resolved_hand_model = Path(hand_model_path or bundle.get("hand_landmarker_model") or DEFAULT_HAND_LANDMARKER_MODEL)
        self.detector = create_hand_landmarker(model_path=resolved_hand_model)

    def close(self) -> None:
        self.detector.close()

    def predict_image_bgr(self, image_bgr: np.ndarray) -> dict[str, object]:
        feat, info = extract_mediapipe_landmarks_feature_from_bgr(
            image_bgr,
            detector=self.detector,
            border_pad_ratio=self.border_pad_ratio,
        )
        if feat is None:
            return {"hand_detected": False, "predicted_label": None, "info": info}

        X = feat.reshape(1, -1)
        pred_idx = int(self.pipeline.predict(X)[0])
        clf = self.pipeline.named_steps["classifier"]
        classes = np.asarray(clf.classes_, dtype=np.int64)
        pred_label = str(self.class_names[pred_idx])
        return {
            "hand_detected": True,
            "predicted_label": pred_label,
            "predicted_class_index": pred_idx,
            "feature_schema": LANDMARK_FEATURE_SCHEMA,
            "feature_count": LANDMARK_FEATURE_DIM,
            "class_count": len(classes),
            "info": info,
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image_path", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hand-model", type=Path, default=None)
    args = p.parse_args()

    bgr = cv2.imread(str(args.image_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image_path}")
    predictor = LandmarkSVMPredictor(args.model, hand_model_path=args.hand_model)
    try:
        res = predictor.predict_image_bgr(bgr)
    finally:
        predictor.close()
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
