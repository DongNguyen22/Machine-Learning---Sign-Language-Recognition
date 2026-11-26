"""Predict an ASL sign (0-9, A-Z) for a single image using the trained LinearSVC model.

Key design guarantees:
- Reuses the exact preprocessing and HOG extraction functions (`extract_hog`,
  `resize_with_padding`, `read_image_bgr`) from `src.preprocess.build_hog_dataset`
  to avoid code duplication or preprocessing drift.
- Loads the model bundle (`models/svm/svm_final_c0_01.joblib` by default) once via
  `SVMSignPredictor`, which can be reused across multiple predictions.
- Maps predictions and top-k scores strictly through `classifier.classes_` and the
  bundle's `class_names` array rather than hardcoding class order.
- Reports linear SVM `decision_score` values directly without converting them to
  pseudo-probabilities or applying softmax.
- Separates image read + preprocessing + HOG extraction time from SVM prediction time.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_hog_dataset import (
    apply_webcam_method_3b_from_bgr,
    extract_hog,
    extract_hog_from_bgr,
    extract_hog_with_diagnostics_from_bgr,
    read_image_bgr,
    resize_with_padding,
)

DEFAULT_MODEL_PATH = Path("models/svm/svm_final_c0_01.joblib")
EXPECTED_HOG_PARAMETERS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
EXPECTED_IMAGE_SIZE = (64, 64)

DECISION_SCORE_EXPLANATION = (
    "decision_score là giá trị hàm quyết định tuyến tính (khoảng cách đại số có dấu "
    "tới siêu phẳng phân lớp của LinearSVC), có thể âm hoặc dương. Đây KHÔNG phải là "
    "xác suất hay độ tin cậy phần trăm (không dùng softmax)."
)
CLOSED_SET_LIMITATION_NOTE = (
    "Mô hình LinearSVC luôn chọn lớp có decision_score cao nhất trong 36 lớp đã học "
    "(0–9, A–Z). Ảnh không có bàn tay, ký hiệu ngoài tập 36 lớp, hoặc ảnh có nền/ánh sáng "
    "khác biệt lớn vẫn sẽ bị gán vào 1 trong 36 nhãn. Chưa đặt ngưỡng từ chối (rejection "
    "threshold) vì chưa có tập dữ liệu ngoài phân phối để kiểm chứng ngưỡng."
)


class SVMSignPredictor:
    """Reusable predictor that loads the saved SVM bundle once and predicts on new images."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file model tại đường dẫn: {self.model_path}"
            )
        if not self.model_path.is_file():
            raise ValueError(
                f"Đường dẫn model không phải là một file hợp lệ: {self.model_path}"
            )

        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:
            raise ValueError(
                f"Không thể nạp file model '{self.model_path}': {exc}"
            ) from exc

        if not isinstance(bundle, dict) or "model" not in bundle or "class_names" not in bundle:
            raise ValueError(
                f"File model '{self.model_path}' thiếu khóa bắt buộc ('model', 'class_names')."
            )

        self.bundle = bundle
        self.pipeline = bundle["model"]
        self.class_names = np.asarray(bundle["class_names"], dtype=str)

        if hasattr(self.pipeline, "named_steps") and "classifier" in self.pipeline.named_steps:
            self.classifier = self.pipeline.named_steps["classifier"]
        else:
            self.classifier = self.pipeline

        if not hasattr(self.classifier, "classes_") or not hasattr(
            self.classifier, "n_features_in_"
        ):
            raise ValueError(
                "Model trong bundle chưa được huấn luyện (thiếu classes_ hoặc n_features_in_)."
            )

        self.model_classes = np.asarray(self.classifier.classes_, dtype=np.int64)
        self.n_features_in = int(self.classifier.n_features_in_)

        if len(self.model_classes) != len(self.class_names):
            raise ValueError(
                f"Số lượng model.classes_ ({len(self.model_classes)}) không khớp với "
                f"class_names ({len(self.class_names)})."
            )
        if np.any(self.model_classes < 0) or np.any(self.model_classes >= len(self.class_names)):
            raise ValueError(
                f"model.classes_ chứa chỉ số nằm ngoài phạm vi class_names: {self.model_classes.tolist()}"
            )

        # Ordered label names aligned strictly with self.classifier.classes_
        self.ordered_class_labels: list[str] = [
            str(self.class_names[int(cls_idx)]) for cls_idx in self.model_classes
        ]

        # Verify saved preprocessing config matches the shared extraction function
        saved_img_size = tuple(bundle.get("image_size", EXPECTED_IMAGE_SIZE))
        if saved_img_size != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                f"Kích thước ảnh của model ({saved_img_size}) không khớp với pipeline ({EXPECTED_IMAGE_SIZE})."
            )

    def validate_feature_vector(self, features: np.ndarray) -> np.ndarray:
        """Validate a 1D (or 1xD) HOG feature vector against classifier.n_features_in_."""
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr_1d = arr[0]
        elif arr.ndim == 1:
            arr_1d = arr
        else:
            raise ValueError(
                f"Vector đặc trưng phải là mảng 1 chiều ({self.n_features_in},) "
                f"hoặc 2 chiều (1, {self.n_features_in}), nhưng nhận được shape={arr.shape}."
            )

        if arr_1d.shape[0] != self.n_features_in:
            raise ValueError(
                f"Số lượng đặc trưng HOG không khớp với model: model yêu cầu "
                f"{self.n_features_in} đặc trưng, nhưng vector đầu vào có {arr_1d.shape[0]} đặc trưng."
            )

        if not np.isfinite(arr_1d).all():
            raise ValueError("Vector đặc trưng HOG chứa giá trị NaN hoặc vô cực (Inf).")

        return arr_1d

    def predict_from_features(
        self,
        features: np.ndarray,
        top_k: int = 3,
    ) -> dict[str, object]:
        """Run LinearSVC inference on a validated 1764-dim HOG feature vector."""
        if top_k <= 0:
            raise ValueError(f"top_k phải lớn hơn 0, nhưng nhận được {top_k}.")

        features_1d = self.validate_feature_vector(features)
        features_2d = features_1d.reshape(1, -1)

        pred_start = time.perf_counter()
        raw_scores = np.asarray(self.pipeline.decision_function(features_2d)[0], dtype=np.float64)
        pred_class_id = int(self.pipeline.predict(features_2d)[0])
        svm_predict_seconds = time.perf_counter() - pred_start

        if raw_scores.shape[0] != len(self.model_classes):
            raise RuntimeError(
                f"Số lượng decision_score ({raw_scores.shape[0]}) không khớp số lớp ({len(self.model_classes)})."
            )

        # Sort positions by decision_score descending
        ranked_positions = np.argsort(raw_scores)[::-1]
        top_pos = int(ranked_positions[0])
        top_class_id = int(self.model_classes[top_pos])
        if top_class_id != pred_class_id:
            raise RuntimeError(
                f"Lớp có decision_score cao nhất ({top_class_id}) không khớp với pipeline.predict ({pred_class_id})."
            )

        predicted_label = str(self.class_names[top_class_id])
        predicted_score = float(raw_scores[top_pos])
        second_score = float(raw_scores[int(ranked_positions[1])])
        margin_top1_top2 = float(predicted_score - second_score)

        k = min(int(top_k), len(self.model_classes))
        top_k_list: list[dict[str, object]] = []
        for rank_idx, pos in enumerate(ranked_positions[:k], start=1):
            pos_int = int(pos)
            cls_id = int(self.model_classes[pos_int])
            top_k_list.append(
                {
                    "rank": rank_idx,
                    "class_index": cls_id,
                    "label": str(self.class_names[cls_id]),
                    "decision_score": float(raw_scores[pos_int]),
                }
            )

        return {
            "predicted_label": predicted_label,
            "predicted_class_index": top_class_id,
            "predicted_decision_score": predicted_score,
            "margin_top1_minus_top2": margin_top1_top2,
            "top_k_predictions": top_k_list,
            "svm_predict_seconds": float(svm_predict_seconds),
            "svm_predict_ms": float(svm_predict_seconds * 1000.0),
        }

    def predict_image(
        self,
        image_path: Path | str,
        top_k: int = 3,
        return_features: bool = False,
    ) -> dict[str, object]:
        """Read an image, extract 1764 HOG features via shared pipeline, and predict its ASL sign."""
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file ảnh tại đường dẫn: {path_obj}"
            )
        if not path_obj.is_file():
            raise ValueError(
                f"Đường dẫn truyền vào không phải là file ảnh: {path_obj}"
            )

        # Stage 1: Image read + resize_with_padding (64x64) + grayscale + HOG (1764 float32)
        hog_start = time.perf_counter()
        hog_features = extract_hog(path_obj)
        preprocess_and_hog_seconds = time.perf_counter() - hog_start

        if hog_features is None:
            raise ValueError(
                f"Không đọc được ảnh hoặc file ảnh bị hỏng/không đúng định dạng: {path_obj}"
            )

        # Stage 2: LinearSVC prediction from the 1764-dim HOG vector in RAM
        pred_result = self.predict_from_features(hog_features, top_k=top_k)
        total_seconds = preprocess_and_hog_seconds + float(pred_result["svm_predict_seconds"])

        result: dict[str, object] = {
            "image_path": str(path_obj),
            "image_path_resolved": str(path_obj.resolve()),
            "model_path": str(self.model_path),
            "predicted_label": pred_result["predicted_label"],
            "predicted_class_index": pred_result["predicted_class_index"],
            "predicted_decision_score": pred_result["predicted_decision_score"],
            "margin_top1_minus_top2": pred_result["margin_top1_minus_top2"],
            "top_k_predictions": pred_result["top_k_predictions"],
            "timing": {
                "preprocess_and_hog_seconds": float(preprocess_and_hog_seconds),
                "preprocess_and_hog_ms": float(preprocess_and_hog_seconds * 1000.0),
                "svm_predict_seconds": pred_result["svm_predict_seconds"],
                "svm_predict_ms": pred_result["svm_predict_ms"],
                "total_end_to_end_seconds": float(total_seconds),
                "total_end_to_end_ms": float(total_seconds * 1000.0),
                "timing_scope": {
                    "preprocess_and_hog": (
                        "Bao gồm đọc file ảnh từ ổ đĩa, resize giữ tỉ lệ + padding trắng 64x64, "
                        "chuyển ảnh xám (grayscale) và trích xuất 1764 đặc trưng HOG (float32)."
                    ),
                    "svm_predict": (
                        "Chỉ bao gồm bước tính decision_function và chọn nhãn của LinearSVC "
                        "từ vector HOG 1764 chiều đã nằm trong RAM (không tính thời gian nạp model)."
                    ),
                },
            },
            "model_metadata": {
                "model_type": type(self.classifier).__name__,
                "c": float(self.bundle.get("c", getattr(self.classifier, "C", 0.01))),
                "feature_count": self.n_features_in,
                "class_count": len(self.model_classes),
                "image_size": list(EXPECTED_IMAGE_SIZE),
                "hog_parameters": {
                    "orientations": EXPECTED_HOG_PARAMETERS["orientations"],
                    "pixels_per_cell": list(EXPECTED_HOG_PARAMETERS["pixels_per_cell"]),
                    "cells_per_block": list(EXPECTED_HOG_PARAMETERS["cells_per_block"]),
                    "block_norm": EXPECTED_HOG_PARAMETERS["block_norm"],
                },
            },
            "decision_score_note": DECISION_SCORE_EXPLANATION,
            "limitation_note": CLOSED_SET_LIMITATION_NOTE,
        }
        if return_features:
            result["hog_features"] = hog_features
        return result

    @staticmethod
    def decode_image_bytes_bgr(image_bytes: bytes) -> np.ndarray:
        """Decode raw JPEG/PNG/BMP bytes in memory into a BGR uint8 numpy array."""
        if not image_bytes:
            raise ValueError("Dữ liệu byte của ảnh rỗng.")
        byte_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(byte_arr, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Không thể giải mã dữ liệu ảnh (ảnh bị hỏng hoặc định dạng không hợp lệ).")
        return image_bgr

    @staticmethod
    def crop_region_bgr(
        image_bgr: np.ndarray,
        crop_box: dict[str, float | int] | None,
    ) -> tuple[np.ndarray, dict[str, int] | None]:
        """Safely crop a region [x, y, width, height] from an in-memory BGR image with bounds clamping."""
        if crop_box is None:
            return image_bgr, None

        img_h, img_w = image_bgr.shape[:2]
        try:
            raw_x = float(crop_box["x"])
            raw_y = float(crop_box["y"])
            raw_w = float(crop_box["width"])
            raw_h = float(crop_box["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "crop_box phải chứa các trường số x, y, width, height."
            ) from exc

        if raw_w <= 1 or raw_h <= 1:
            raise ValueError(
                f"Kích thước vùng cắt không hợp lệ: width={raw_w}, height={raw_h}."
            )

        x1 = max(0, min(img_w - 1, int(np.floor(raw_x))))
        y1 = max(0, min(img_h - 1, int(np.floor(raw_y))))
        x2 = max(x1 + 1, min(img_w, int(np.ceil(raw_x + raw_w))))
        y2 = max(y1 + 1, min(img_h, int(np.ceil(raw_y + raw_h))))

        cropped = image_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            raise ValueError("Vùng ảnh sau khi cắt rỗng.")

        applied = {
            "x": int(x1),
            "y": int(y1),
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "source_width": int(img_w),
            "source_height": int(img_h),
        }
        return cropped, applied

    @staticmethod
    def encode_png_data_url(image: np.ndarray) -> str:
        """Encode an in-memory BGR or grayscale uint8 numpy array as a lossless PNG data URL."""
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Không thể mã hóa ảnh chẩn đoán sang định dạng PNG.")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def predict_image_bgr(
        self,
        image_bgr: np.ndarray,
        top_k: int = 3,
        crop_box: dict[str, float | int] | None = None,
        return_features: bool = False,
        include_diagnostics: bool = False,
        frame_id: str | int | None = None,
        experimental_webcam_mode: bool = False,
        already_flipped: bool = False,
    ) -> dict[str, object]:
        """Predict an ASL sign directly from an in-memory BGR image (no disk I/O).

        When `experimental_webcam_mode=False` (default): uses the legacy pipeline (input crop as-is).
        When `experimental_webcam_mode=True`: applies Method 3B (single horizontal flip if not
        `already_flipped` + `suppress_background_preserve_hand`) before `resize_with_padding` (64x64)
        and 1764-dim HOG extraction.
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            raise ValueError("Ảnh BGR trong bộ nhớ không hợp lệ hoặc rỗng.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                f"Ảnh đầu vào phải có 3 kênh màu BGR (H, W, 3), nhưng nhận được shape={image_bgr.shape}."
            )

        hog_start = time.perf_counter()
        target_bgr, applied_crop = self.crop_region_bgr(image_bgr, crop_box)

        # Reconstruct canonical unflipped raw crop so we never double-flip and can always compare
        # legacy vs Method 3B on the exact same frame.
        raw_unflipped_bgr = cv2.flip(target_bgr, 1) if already_flipped else target_bgr

        if experimental_webcam_mode:
            model_input_bgr, _flipped_once_bgr, _hand_mask = apply_webcam_method_3b_from_bgr(
                target_bgr,
                already_flipped=already_flipped,
            )
            webcam_mode = "method_3b"
            webcam_mode_label = "Cách 3B (Thử nghiệm: Lật ngang + Giảm nền giữ viền)"
        else:
            model_input_bgr = target_bgr
            webcam_mode = "legacy"
            webcam_mode_label = (
                "Cách cũ (Chỉ lật ngang)"
                if already_flipped
                else "Cách cũ (Ảnh crop gốc)"
            )

        diag_tuple = extract_hog_with_diagnostics_from_bgr(model_input_bgr)
        preprocess_and_hog_seconds = time.perf_counter() - hog_start

        if diag_tuple is None:
            raise ValueError("Không thể trích xuất đặc trưng HOG từ ảnh trong bộ nhớ.")

        hog_features, _resized_bgr_64, gray_uint8_64 = diag_tuple

        pred_result = self.predict_from_features(hog_features, top_k=top_k)
        total_seconds = preprocess_and_hog_seconds + float(pred_result["svm_predict_seconds"])

        orig_h, orig_w = image_bgr.shape[:2]
        proc_h, proc_w = target_bgr.shape[:2]

        result: dict[str, object] = {
            "frame_id": frame_id,
            "webcam_mode": webcam_mode,
            "webcam_mode_label": webcam_mode_label,
            "experimental_webcam_mode": bool(experimental_webcam_mode),
            "already_flipped": bool(already_flipped),
            "label": pred_result["predicted_label"],
            "predicted_label": pred_result["predicted_label"],
            "predicted_class_index": pred_result["predicted_class_index"],
            "decision_score": pred_result["predicted_decision_score"],
            "predicted_decision_score": pred_result["predicted_decision_score"],
            "margin_top1_minus_top2": pred_result["margin_top1_minus_top2"],
            "score_type": "decision_score",
            "probability_calibrated": False,
            "top_k": pred_result["top_k_predictions"],
            "top_k_predictions": pred_result["top_k_predictions"],
            "timing": {
                "preprocess_and_hog_seconds": float(preprocess_and_hog_seconds),
                "preprocess_and_hog_ms": float(preprocess_and_hog_seconds * 1000.0),
                "svm_predict_seconds": pred_result["svm_predict_seconds"],
                "svm_predict_ms": pred_result["svm_predict_ms"],
                "total_end_to_end_seconds": float(total_seconds),
                "total_end_to_end_ms": float(total_seconds * 1000.0),
            },
            "image_dimensions": {
                "original_width": int(orig_w),
                "original_height": int(orig_h),
                "processed_width": int(proc_w),
                "processed_height": int(proc_h),
            },
            "crop_applied": applied_crop,
            "decision_score_note": DECISION_SCORE_EXPLANATION,
            "limitation_note": CLOSED_SET_LIMITATION_NOTE,
        }
        if include_diagnostics:
            # Compute both Legacy (unflipped raw crop) and Method 3B on the exact same frame
            # so the user can inspect and save side-by-side results on the same frame.
            legacy_tuple = extract_hog_with_diagnostics_from_bgr(raw_unflipped_bgr)
            if legacy_tuple is None:
                raise ValueError("Không thể trích xuất HOG cho chế độ Cách cũ trên khung hình này.")
            legacy_hog, _legacy_bgr64, legacy_gray64 = legacy_tuple
            legacy_pred = self.predict_from_features(legacy_hog, top_k=top_k)

            m3b_input_bgr, _m3b_flipped_bgr, _m3b_mask = apply_webcam_method_3b_from_bgr(
                raw_unflipped_bgr,
                already_flipped=False,
            )
            m3b_tuple = extract_hog_with_diagnostics_from_bgr(m3b_input_bgr)
            if m3b_tuple is None:
                raise ValueError("Không thể trích xuất HOG cho Cách 3B trên khung hình này.")
            m3b_hog, _m3b_bgr64, m3b_gray64 = m3b_tuple
            m3b_pred = self.predict_from_features(m3b_hog, top_k=top_k)

            same_frame_comparison = {
                "legacy": {
                    "mode": "legacy",
                    "mode_label": "Cách cũ (Ảnh crop gốc)",
                    "predicted_label": legacy_pred["predicted_label"],
                    "predicted_decision_score": legacy_pred["predicted_decision_score"],
                    "margin_top1_minus_top2": legacy_pred["margin_top1_minus_top2"],
                    "top_k": legacy_pred["top_k_predictions"],
                    "preprocessed_gray_64x64_png_data_url": self.encode_png_data_url(legacy_gray64),
                },
                "method_3b": {
                    "mode": "method_3b",
                    "mode_label": "Cách 3B (Lật ngang + Giảm nền giữ viền)",
                    "predicted_label": m3b_pred["predicted_label"],
                    "predicted_decision_score": m3b_pred["predicted_decision_score"],
                    "margin_top1_minus_top2": m3b_pred["margin_top1_minus_top2"],
                    "top_k": m3b_pred["top_k_predictions"],
                    "preprocessed_gray_64x64_png_data_url": self.encode_png_data_url(m3b_gray64),
                },
            }

            result["same_frame_comparison"] = same_frame_comparison
            result["diagnostics"] = {
                "frame_id": frame_id,
                "webcam_mode": webcam_mode,
                "webcam_mode_label": webcam_mode_label,
                "experimental_webcam_mode": bool(experimental_webcam_mode),
                "already_flipped": bool(already_flipped),
                "received_crop_png_data_url": self.encode_png_data_url(raw_unflipped_bgr),
                "model_input_crop_png_data_url": self.encode_png_data_url(model_input_bgr),
                "preprocessed_gray_64x64_png_data_url": self.encode_png_data_url(gray_uint8_64),
                "received_crop_shape": [int(proc_h), int(proc_w), 3],
                "gray_64x64_shape": [int(gray_uint8_64.shape[0]), int(gray_uint8_64.shape[1])],
                "gray_64x64_min": int(np.min(gray_uint8_64)),
                "gray_64x64_max": int(np.max(gray_uint8_64)),
                "gray_64x64_mean": float(np.mean(gray_uint8_64)),
                "hog_feature_count": int(hog_features.shape[0]),
                "hog_l2_norm": float(np.linalg.norm(hog_features)),
                "hog_first_10": [float(v) for v in hog_features[:10]],
                "same_frame_comparison": same_frame_comparison,
            }
        if return_features:
            result["hog_features"] = hog_features
        return result

    def predict_image_bytes(
        self,
        image_bytes: bytes,
        top_k: int = 3,
        crop_box: dict[str, float | int] | None = None,
        return_features: bool = False,
        include_diagnostics: bool = False,
        frame_id: str | int | None = None,
        experimental_webcam_mode: bool = False,
        already_flipped: bool = False,
    ) -> dict[str, object]:
        """Decode image bytes in RAM, extract 1764 HOG features, and run LinearSVC prediction."""
        decode_start = time.perf_counter()
        image_bgr = self.decode_image_bytes_bgr(image_bytes)
        decode_seconds = time.perf_counter() - decode_start

        result = self.predict_image_bgr(
            image_bgr,
            top_k=top_k,
            crop_box=crop_box,
            return_features=return_features,
            include_diagnostics=include_diagnostics,
            frame_id=frame_id,
            experimental_webcam_mode=experimental_webcam_mode,
            already_flipped=already_flipped,
        )
        timing = dict(result["timing"])  # type: ignore[arg-type]
        preprocess_total_s = decode_seconds + float(timing["preprocess_and_hog_seconds"])
        total_s = preprocess_total_s + float(timing["svm_predict_seconds"])
        timing["image_decode_ms"] = float(decode_seconds * 1000.0)
        timing["preprocess_and_hog_seconds"] = float(preprocess_total_s)
        timing["preprocess_and_hog_ms"] = float(preprocess_total_s * 1000.0)
        timing["total_end_to_end_seconds"] = float(total_s)
        timing["total_end_to_end_ms"] = float(total_s * 1000.0)
        result["timing"] = timing
        return result


def format_terminal_report(result: dict[str, object]) -> str:
    timing = result["timing"]
    top_k_preds = result["top_k_predictions"]
    lines = [
        "================ KẾT QUẢ DỰ ĐOÁN KÝ HIỆU ASL (SVM) ================",
        f"Ảnh đầu vào          : {result['image_path']}",
        f"Model sử dụng        : {result['model_path']}",
        f"Ký hiệu dự đoán      : {result['predicted_label']} (class_index={result['predicted_class_index']})",
        f"Decision score (#1)  : {result['predicted_decision_score']:+.6f} "
        f"(chênh lệch so với #2: {result['margin_top1_minus_top2']:.6f})",
        "",
        f"--- Top {len(top_k_preds)} lớp có decision_score cao nhất ---",
    ]
    for item in top_k_preds:
        lines.append(
            f"  #{item['rank']}: Ký hiệu '{item['label']}' (class_index={item['class_index']:2d}) "
            f"| decision_score = {item['decision_score']:+.6f}"
        )
    lines.extend(
        [
            "",
            "--- Thời gian xử lý (không tính thời gian nạp model một lần) ---",
            f"  1. Đọc ảnh + Tiền xử lý 64x64 + Trích xuất HOG (1764 chiều): {timing['preprocess_and_hog_ms']:.3f} ms "
            f"({timing['preprocess_and_hog_seconds']:.6f} s)",
            f"  2. Dự đoán LinearSVC từ vector HOG trong RAM               : {timing['svm_predict_ms']:.3f} ms "
            f"({timing['svm_predict_seconds']:.6f} s)",
            f"  => Tổng thời gian xử lý toàn bộ 1 ảnh                      : {timing['total_end_to_end_ms']:.3f} ms "
            f"({timing['total_end_to_end_seconds']:.6f} s)",
            "",
            "--- Lưu ý giải thích ---",
            f"  * {result['decision_score_note']}",
            f"  * {result['limitation_note']}",
            "===================================================================",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán ký hiệu ngôn ngữ tay ASL (0-9, A-Z) cho một ảnh bằng mô hình LinearSVC đã lưu."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=None,
        help="Đường dẫn tới file ảnh cần dự đoán (.jpg, .jpeg, .png, .bmp)",
    )
    parser.add_argument(
        "--image",
        dest="image_opt",
        type=Path,
        default=None,
        help="Đường dẫn tới file ảnh cần dự đoán (thay thế cho tham số vị trí)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Đường dẫn tới file model joblib (mặc định: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số lượng lớp có decision_score cao nhất cần hiển thị (mặc định: 3)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="In kết quả ra màn hình dưới định dạng JSON",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Đường dẫn file để lưu kết quả dự đoán dạng JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Ensure UTF-8 terminal output on Windows PowerShell
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    target_image = args.image_opt if args.image_opt is not None else args.image_path
    if target_image is None:
        print(
            "Lỗi: Vui lòng cung cấp đường dẫn ảnh cần dự đoán (ví dụ: python src/model/predict_svm.py \"duong_dan_anh.jpg\").",
            file=sys.stderr,
        )
        return 2

    try:
        predictor = SVMSignPredictor(model_path=args.model)
        result = predictor.predict_image(target_image, top_k=args.top_k)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Lỗi khi dự đoán: {exc}", file=sys.stderr)
        return 1

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(format_terminal_report(result))
        if args.json_output is not None:
            print(f"Đã lưu kết quả JSON tại: {args.json_output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
