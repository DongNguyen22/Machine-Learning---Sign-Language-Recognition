"""Read-only inference wrapper for Landmark SVM v2/v3 (86-dim) across 3 scopes:
- `all_36`     : 36 classes (0–9, A–Z) from `models/svm/experiments/landmark_asl_v2/svm_landmark_asl_v2.joblib`
- `letters_az` : 26 classes (A–Z)      from `models/svm/experiments/landmark_asl_v3_split/svm_landmark_az_v3.joblib`
- `digits_09`  : 10 classes (0–9)      from `models/svm/experiments/landmark_asl_v3_split/svm_landmark_09_v3.joblib`

Guarantees:
- Uses the exact `normalize_hand_landmarks_geom_v2(landmarks, image_width, image_height)`
  function from `src/preprocess/webcam_features.py` (zero formula duplication).
- Strictly validates 21 finite (x, y) landmarks and positive finite (image_width, image_height).
- Preserves exact `pipeline.predict()` label and exact `pipeline.decision_function()` values,
  while computing a monotonic heuristic `confidence` in (0, 1) and relative Top-K weights
  from sklearn's OvR vote counts and recovered OvO cumulative margins (validated on
  `validation (P8, P10)` across all 3 scopes with 0.00% saturation > 0.99).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.webcam_features import (
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    normalize_hand_landmarks_geom_v2,
)

DEFAULT_LANDMARK_V2_MODEL_PATH = (
    PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v2" / "svm_landmark_asl_v2.joblib"
)
DEFAULT_LANDMARK_AZ_MODEL_PATH = (
    PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v3_split" / "svm_landmark_az_v3.joblib"
)
DEFAULT_LANDMARK_09_MODEL_PATH = (
    PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v3_split" / "svm_landmark_09_v3.joblib"
)

VALID_LANDMARK_SCOPES = ("all_36", "letters_az", "digits_09")
SCOPE_DISPLAY_LABELS = {
    "all_36": "Tất cả 36",
    "letters_az": "Chữ cái A–Z",
    "digits_09": "Chữ số 0–9",
}
DEFAULT_SCOPE_MODEL_PATHS: dict[str, Path] = {
    "all_36": DEFAULT_LANDMARK_V2_MODEL_PATH,
    "letters_az": DEFAULT_LANDMARK_AZ_MODEL_PATH,
    "digits_09": DEFAULT_LANDMARK_09_MODEL_PATH,
}

CONFIDENCE_NOTE = (
    "confidence là điểm tin cậy ước lượng đơn điệu trong (0, 1) suy từ số phiếu OvO và "
    "chênh lệch biên cộng dồn của SVC(kernel='rbf') OvR (không phải xác suất hiệu chuẩn Platt)."
)


def _is_finite_real(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def recover_ovr_votes_and_ovo_margin(ovr_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert sklearn SVC OvR score `v + conf / (3 * (|conf| + 1))` into (votes, sum_conf)."""
    scores = np.asarray(ovr_scores, dtype=np.float64)
    votes = np.rint(scores)
    frac = scores - votes
    abs_3f = np.minimum(3.0 * np.abs(frac), 1.0 - 1e-6)
    sum_conf = np.sign(frac) * (abs_3f / (1.0 - abs_3f))
    return votes, sum_conf


def compute_ovr_heuristic_confidence(
    ovr_scores: np.ndarray,
    top_positions: np.ndarray,
    n_classes: int,
) -> tuple[float, list[float], np.ndarray]:
    """Compute heuristic confidence in (0, 1) and Top-K relative weights from OvR scores.

    Validated on `validation (P8, P10)` across all 3 scopes (`all_36`, `letters_az`, `digits_09`):
    - `vote_gap = votes[top1] - votes[top2]`
    - `delta_avg = (sum_conf[top1] - sum_conf[top2]) / max(1, n_classes - 1)`
    - `confidence = sigmoid(0.8 * vote_gap + 3.5 * delta_avg - 0.5)`
      (0.00% saturated > 0.99 across all 3 scopes; strong correct-vs-wrong separation).
    """
    votes, sum_conf = recover_ovr_votes_and_ovo_margin(ovr_scores)
    denom = float(max(1, n_classes - 1))

    if len(top_positions) >= 2:
        p1, p2 = int(top_positions[0]), int(top_positions[1])
        vote_gap = float(votes[p1] - votes[p2])
        delta_avg = float((sum_conf[p1] - sum_conf[p2]) / denom)
    else:
        vote_gap = 1.0
        delta_avg = 0.0

    logit = max(-20.0, min(20.0, 0.8 * vote_gap + 3.5 * delta_avg - 0.5))
    confidence = float(1.0 / (1.0 + math.exp(-logit)))

    strengths = np.array(
        [(votes[int(p)] * 0.25 + sum_conf[int(p)] / denom) / 0.25 for p in top_positions],
        dtype=np.float64,
    )
    strengths -= np.max(strengths)
    exps = np.exp(strengths)
    weights = (exps / np.sum(exps)).tolist()
    return confidence, [float(w) for w in weights], votes


class LandmarkV2SignPredictor:
    """Read-only predictor for an 86-dim Geometric Landmark SVM bundle (36, 26, or 10 classes)."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_LANDMARK_V2_MODEL_PATH,
        metadata_path: Path | str | None = None,
        scope: str = "all_36",
    ) -> None:
        self.scope = scope
        self.scope_label = SCOPE_DISPLAY_LABELS.get(scope, scope)
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy model Landmark SVM ({scope}) tại: {self.model_path}")

        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict) or "model" not in bundle or "class_names" not in bundle:
            raise ValueError(f"Bundle Landmark SVM ({scope}) không hợp lệ: {self.model_path}")

        if bundle.get("feature_schema") != LANDMARK_GEOM_V2_FEATURE_SCHEMA:
            raise ValueError(
                f"Schema model Landmark ({scope}) không khớp: cần '{LANDMARK_GEOM_V2_FEATURE_SCHEMA}', "
                f"nhận được {bundle.get('feature_schema')!r}."
            )
        if int(bundle.get("feature_count", -1)) != LANDMARK_GEOM_V2_FEATURE_DIM:
            raise ValueError(
                f"Số chiều đặc trưng Landmark ({scope}) không khớp: cần {LANDMARK_GEOM_V2_FEATURE_DIM}, "
                f"nhận được {bundle.get('feature_count')!r}."
            )

        pipe = bundle["model"]
        if not isinstance(pipe, Pipeline) or "scaler" not in pipe.named_steps or "clf" not in pipe.named_steps:
            raise ValueError("Model Landmark SVM phải là Pipeline([('scaler', ...), ('clf', ...)]).")
        if not isinstance(pipe.named_steps["scaler"], StandardScaler) or not isinstance(
            pipe.named_steps["clf"], (SVC, LinearSVC)
        ):
            raise ValueError("Model Landmark SVM phải dùng StandardScaler + SVC/LinearSVC.")

        self.bundle = bundle
        self.pipeline: Pipeline = pipe
        self.classifier = pipe.named_steps["clf"]
        self.class_names = np.asarray(bundle["class_names"], dtype=str)
        self.model_classes = np.asarray(self.classifier.classes_, dtype=np.int64)
        self.n_features_in = int(self.classifier.n_features_in_)
        if self.n_features_in != LANDMARK_GEOM_V2_FEATURE_DIM:
            raise ValueError(
                f"classifier.n_features_in_={self.n_features_in} != {LANDMARK_GEOM_V2_FEATURE_DIM}"
            )
        self.ordered_class_labels: list[str] = [
            str(self.class_names[int(cid)]) for cid in self.model_classes
        ]

        resolved_meta = (
            Path(metadata_path)
            if metadata_path is not None
            else self.model_path.with_name(f"{self.model_path.stem}_metadata.json")
        )
        self.metadata_path = resolved_meta if resolved_meta.is_file() else None
        self.metadata: dict[str, Any] = (
            json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if self.metadata_path is not None
            else {}
        )

    @staticmethod
    def validate_landmarks_input(
        landmarks: Any,
        image_width: Any,
        image_height: Any,
    ) -> tuple[list[dict[str, float]], float, float]:
        if not isinstance(landmarks, list) or len(landmarks) != 21:
            got = len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__
            raise ValueError(f"Trường 'landmarks' phải là mảng đúng 21 điểm mốc (nhận được: {got}).")

        if not _is_finite_real(image_width) or float(image_width) <= 0:
            raise ValueError(f"Trường 'image_width' phải là số hữu hạn > 0 (nhận được: {image_width!r}).")
        if not _is_finite_real(image_height) or float(image_height) <= 0:
            raise ValueError(f"Trường 'image_height' phải là số hữu hạn > 0 (nhận được: {image_height!r}).")

        cleaned: list[dict[str, float]] = []
        for idx, pt in enumerate(landmarks):
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                raise ValueError(f"Điểm mốc #{idx} phải là object có trường 'x' và 'y'.")
            if not _is_finite_real(pt["x"]) or not _is_finite_real(pt["y"]):
                raise ValueError(f"Điểm mốc #{idx} chứa tọa độ không hữu hạn (NaN/Inf hoặc sai kiểu).")
            cleaned.append({"x": float(pt["x"]), "y": float(pt["y"])})

        return cleaned, float(image_width), float(image_height)

    def predict_from_landmarks(
        self,
        landmarks: Any,
        image_width: Any,
        image_height: Any,
        top_k: int = 3,
        frame_id: str | int | None = None,
    ) -> dict[str, object]:
        if top_k <= 0 or top_k > 36:
            raise ValueError("Tham số 'top_k' phải nằm trong [1, 36].")

        cleaned_pts, w, h = self.validate_landmarks_input(landmarks, image_width, image_height)

        t_feat0 = time.perf_counter()
        feat_86 = normalize_hand_landmarks_geom_v2(cleaned_pts, image_width=w, image_height=h)
        feat_s = time.perf_counter() - t_feat0

        X = feat_86.reshape(1, -1)
        t_pred0 = time.perf_counter()
        pred_class_id = int(self.pipeline.predict(X)[0])
        raw_scores = np.asarray(self.pipeline.decision_function(X)[0], dtype=np.float64)
        pred_s = time.perf_counter() - t_pred0

        ranked_positions = np.argsort(raw_scores)[::-1]
        pred_pos_matches = np.where(self.model_classes == pred_class_id)[0]
        if len(pred_pos_matches) != 1:
            raise RuntimeError(f"Không tìm thấy lớp dự đoán {pred_class_id} trong model.classes_.")
        pred_pos = int(pred_pos_matches[0])

        if int(ranked_positions[0]) != pred_pos:
            ordered_positions = np.concatenate(
                [[pred_pos], ranked_positions[ranked_positions != pred_pos]]
            )
        else:
            ordered_positions = ranked_positions

        k = min(int(top_k), len(self.model_classes))
        top_positions = ordered_positions[:k]
        confidence, rel_weights, votes = compute_ovr_heuristic_confidence(
            raw_scores, top_positions, len(self.model_classes)
        )

        pred_label = str(self.class_names[pred_class_id])
        pred_score = float(raw_scores[pred_pos])
        second_score = float(raw_scores[int(ordered_positions[1])]) if len(ordered_positions) >= 2 else pred_score
        margin_top1_top2 = float(pred_score - second_score)

        top_k_list: list[dict[str, object]] = []
        for rank_idx, (pos, rw) in enumerate(zip(top_positions, rel_weights), start=1):
            pos_int = int(pos)
            cls_id = int(self.model_classes[pos_int])
            top_k_list.append(
                {
                    "rank": rank_idx,
                    "class_index": cls_id,
                    "label": str(self.class_names[cls_id]),
                    "decision_score": float(raw_scores[pos_int]),
                    "ovr_votes": int(votes[pos_int]),
                    "relative_weight": round(float(rw), 4),
                }
            )

        total_s = feat_s + pred_s
        return {
            "frame_id": frame_id,
            "scope": self.scope,
            "scope_label": self.scope_label,
            "model_mode": "landmark_v2",
            "webcam_mode": "landmark_v2",
            "webcam_mode_label": f"Landmark SVM ({self.scope_label})",
            "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
            "feature_count": LANDMARK_GEOM_V2_FEATURE_DIM,
            "class_count": len(self.model_classes),
            "label": pred_label,
            "predicted_label": pred_label,
            "predicted_class_index": pred_class_id,
            "decision_score": pred_score,
            "predicted_decision_score": pred_score,
            "margin_top1_minus_top2": margin_top1_top2,
            "confidence": round(confidence, 4),
            "score_type": "ovr_decision_score_heuristic_confidence",
            "probability_calibrated": False,
            "confidence_note": CONFIDENCE_NOTE,
            "top_k": top_k_list,
            "top_k_predictions": top_k_list,
            "input_dimensions": {"image_width": w, "image_height": h},
            "timing": {
                "feature_extract_ms": float(feat_s * 1000.0),
                "preprocess_and_hog_ms": float(feat_s * 1000.0),
                "svm_predict_ms": float(pred_s * 1000.0),
                "total_end_to_end_ms": float(total_s * 1000.0),
            },
        }


class LandmarkV2ScopeRegistry:
    """Loads all 3 Landmark SVM scopes independently so missing one never breaks the others."""

    def __init__(
        self,
        scope_paths: dict[str, Path | str | None] | None = None,
    ) -> None:
        resolved_paths: dict[str, Path | None] = {}
        for scope in VALID_LANDMARK_SCOPES:
            if scope_paths is not None and scope in scope_paths:
                p = scope_paths[scope]
                resolved_paths[scope] = Path(p) if p is not None else None
            else:
                resolved_paths[scope] = DEFAULT_SCOPE_MODEL_PATHS[scope]

        self.scope_paths = resolved_paths
        self.predictors: dict[str, LandmarkV2SignPredictor | None] = {}
        self.errors: dict[str, str | None] = {}

        for scope in VALID_LANDMARK_SCOPES:
            p = self.scope_paths[scope]
            if p is None:
                self.predictors[scope] = None
                self.errors[scope] = f"Phạm vi '{scope}' chưa được cấu hình đường dẫn model."
                continue
            try:
                self.predictors[scope] = LandmarkV2SignPredictor(model_path=p, scope=scope)
                self.errors[scope] = None
            except Exception as exc:
                self.predictors[scope] = None
                self.errors[scope] = str(exc)

    def get_predictor(self, scope: str) -> LandmarkV2SignPredictor | None:
        return self.predictors.get(scope)

    @property
    def any_available(self) -> bool:
        return any(p is not None for p in self.predictors.values())

    def get_scope_health(self, scope: str) -> dict[str, object]:
        pred = self.predictors.get(scope)
        p = self.scope_paths.get(scope)
        if pred is None:
            return {
                "available": False,
                "scope": scope,
                "label": SCOPE_DISPLAY_LABELS.get(scope, scope),
                "model_path": str(p) if p is not None else None,
                "feature_schema": None,
                "feature_dim": None,
                "class_count": None,
                "labels": [],
                "candidate_id": None,
                "error": self.errors.get(scope),
            }
        return {
            "available": True,
            "scope": scope,
            "label": SCOPE_DISPLAY_LABELS.get(scope, scope),
            "model_path": str(p),
            "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA,
            "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM,
            "class_count": len(pred.ordered_class_labels),
            "labels": pred.ordered_class_labels,
            "candidate_id": pred.bundle.get("candidate_id"),
            "error": None,
        }

    def get_health_dict(self) -> dict[str, object]:
        scopes_health = {sc: self.get_scope_health(sc) for sc in VALID_LANDMARK_SCOPES}
        primary = self.predictors.get("all_36")
        primary_health = scopes_health["all_36"]
        return {
            "available": self.any_available,
            "default_scope": "all_36",
            "experimental": False,
            "model_path": primary_health["model_path"],
            "feature_schema": LANDMARK_GEOM_V2_FEATURE_SCHEMA if self.any_available else None,
            "feature_dim": LANDMARK_GEOM_V2_FEATURE_DIM if self.any_available else None,
            "class_count": len(primary.ordered_class_labels) if primary is not None else None,
            "candidate_id": primary.bundle.get("candidate_id") if primary is not None else None,
            "scopes": scopes_health,
            "error": None if self.any_available else self.errors.get("all_36"),
        }
