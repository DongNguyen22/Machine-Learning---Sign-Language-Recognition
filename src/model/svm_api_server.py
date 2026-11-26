"""Minimal HTTP API server for live camera ASL sign recognition using SVMSignPredictor.

Why this architecture:
- Uses Python's built-in `http.server.ThreadingHTTPServer` so it runs directly inside
  the project's `.venv` without requiring extra web frameworks.
- Loads `models/svm/svm_final_c0_01.joblib` ONCE at server startup via `SVMSignPredictor`.
- Processes camera frames completely in memory (base64 -> bytes -> cv2.imdecode ->
  resize_with_padding 64x64 -> grayscale -> 1764 HOG float32 -> LinearSVC), never writing
  camera frames to disk by default.
- Exposes:
  + GET  /health            : Readiness check, model metadata, and the 36 ordered class labels.
  + GET  /labels            : Returns the 36 class labels in exact `model.classes_` order.
  + GET  /reference-samples : Returns reference training samples for classes A, B, C, J
                              (original crop + actual 64x64 grayscale HOG input + CLI scores)
                              for side-by-side diagnostic comparison and frontend flow parity tests.
  + POST /predict           : Accepts `{"image": "<base64 JPEG/PNG>", "top_k": 3, "crop_box": optional,
                              "frame_id": optional, "include_diagnostics": optional}` and returns
                              predicted label, Top 3 `decision_score` values, timings, and optional
                              in-pipeline diagnostic images (`received_crop` + `gray_64x64`).
  + POST /diagnostic/save   : Saves a single diagnostic snapshot to `results/svm/webcam_diagnostics/`
                              ONLY when explicitly triggered by the user clicking "Lưu mẫu chẩn đoán".
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict_landmark_v2 import (
    DEFAULT_LANDMARK_09_MODEL_PATH,
    DEFAULT_LANDMARK_AZ_MODEL_PATH,
    DEFAULT_LANDMARK_V2_MODEL_PATH,
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    LandmarkV2ScopeRegistry,
    VALID_LANDMARK_SCOPES,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor

DEFAULT_HOST = os.environ.get("SVM_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SVM_API_PORT", "8000"))
DEFAULT_MAX_PAYLOAD_BYTES = int(os.environ.get("SVM_API_MAX_BYTES", str(4 * 1024 * 1024)))
DEFAULT_MAX_IMAGE_BYTES = int(os.environ.get("SVM_API_MAX_IMAGE_BYTES", str(3 * 1024 * 1024)))
DEFAULT_LANDMARK_V2_PATH_CONFIG = Path(
    os.environ.get("SVM_LANDMARK_V2_MODEL_PATH", str(DEFAULT_LANDMARK_V2_MODEL_PATH))
)
DEFAULT_LANDMARK_AZ_PATH_CONFIG = Path(
    os.environ.get("SVM_LANDMARK_AZ_MODEL_PATH", str(DEFAULT_LANDMARK_AZ_MODEL_PATH))
)
DEFAULT_LANDMARK_09_PATH_CONFIG = Path(
    os.environ.get("SVM_LANDMARK_09_MODEL_PATH", str(DEFAULT_LANDMARK_09_MODEL_PATH))
)
DEFAULT_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SVM_API_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

DEFAULT_DATASET_PATH = Path(
    r"C:\Users\Quang Diep\Downloads\ASL-HG American Sign Language Hand Gesture Image D"
    r"\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000"
    r"\ASL_Processed_Images\asl_processed"
)
DEFAULT_DIAGNOSTIC_SAVE_DIR = Path(
    os.environ.get("SVM_DIAGNOSTIC_SAVE_DIR", "results/svm/webcam_diagnostics")
)
REFERENCE_CLASSES = ["A", "B", "C", "J"]


class SVMAPIApp:
    """Holds the singleton SVMSignPredictor instance, cached reference samples, and server config."""

    def __init__(
        self,
        model_path: Path,
        allowed_origins: list[str],
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        dataset_path: Path = DEFAULT_DATASET_PATH,
        diagnostic_save_dir: Path = DEFAULT_DIAGNOSTIC_SAVE_DIR,
        landmark_v2_model_path: Path | None = DEFAULT_LANDMARK_V2_PATH_CONFIG,
        landmark_az_model_path: Path | None = DEFAULT_LANDMARK_AZ_PATH_CONFIG,
        landmark_09_model_path: Path | None = DEFAULT_LANDMARK_09_PATH_CONFIG,
    ) -> None:
        self.startup_time = time.time()
        self.model_path = model_path
        self.allowed_origins = set(allowed_origins)
        self.max_payload_bytes = max_payload_bytes
        self.max_image_bytes = max_image_bytes
        self.dataset_path = dataset_path
        self.diagnostic_save_dir = diagnostic_save_dir
        self.predictor = SVMSignPredictor(model_path=model_path)
        self._reference_samples_cache: list[dict[str, object]] | None = None

        # Multi-scope Landmark SVM registry (read-only, independent per scope; never blocks HOG)
        self.landmark_registry = LandmarkV2ScopeRegistry(
            scope_paths={
                "all_36": landmark_v2_model_path,
                "letters_az": landmark_az_model_path,
                "digits_09": landmark_09_model_path,
            }
        )
        self.landmark_v2_model_path = Path(landmark_v2_model_path) if landmark_v2_model_path else None
        self.landmark_v2_predictor = self.landmark_registry.get_predictor("all_36")
        self.landmark_v2_available = self.landmark_registry.any_available
        self.landmark_v2_error = self.landmark_registry.errors.get("all_36")

    def get_landmark_v2_health(self) -> dict[str, object]:
        return self.landmark_registry.get_health_dict()

    def get_reference_samples(self) -> list[dict[str, object]]:
        """Load reference training samples of classes A, B, C, J from asl_processed/train."""
        if self._reference_samples_cache is not None:
            return self._reference_samples_cache

        samples: list[dict[str, object]] = []
        train_dir = self.dataset_path / "train"
        if not train_dir.is_dir():
            self._reference_samples_cache = []
            return []

        for cls_name in REFERENCE_CLASSES:
            cls_dir = train_dir / cls_name
            if not cls_dir.is_dir():
                continue
            files = sorted(cls_dir.glob("*.jpg"))
            if not files:
                continue
            # Pick first file (e.g. P10_A_902.jpg) as canonical reference
            ref_path = files[0]
            raw_bytes = ref_path.read_bytes()
            pred = self.predictor.predict_image_bytes(
                image_bytes=raw_bytes,
                top_k=3,
                include_diagnostics=True,
                frame_id=f"ref_{cls_name}_{ref_path.name}",
            )
            diag = pred.get("diagnostics", {})
            samples.append(
                {
                    "class_name": cls_name,
                    "filename": ref_path.name,
                    "split": "train",
                    "reference_note": (
                        "Ảnh tham chiếu từ tập train (asl_processed), chỉ dùng để đối chiếu "
                        "trực quan và kiểm tra tính nhất quán kỹ thuật của luồng gửi ảnh, "
                        "không phải bộ đánh giá mới."
                    ),
                    "original_jpeg_base64": base64.b64encode(raw_bytes).decode("ascii"),
                    "received_crop_png_data_url": diag.get("received_crop_png_data_url"),
                    "preprocessed_gray_64x64_png_data_url": diag.get(
                        "preprocessed_gray_64x64_png_data_url"
                    ),
                    "image_dimensions": pred.get("image_dimensions"),
                    "cli_predicted_label": pred.get("predicted_label"),
                    "cli_decision_score": pred.get("predicted_decision_score"),
                    "cli_margin_top1_minus_top2": pred.get("margin_top1_minus_top2"),
                    "cli_top_k": pred.get("top_k"),
                    "hog_l2_norm": diag.get("hog_l2_norm"),
                    "gray_64x64_mean": diag.get("gray_64x64_mean"),
                }
            )

        self._reference_samples_cache = samples
        return samples


def create_request_handler(app: SVMAPIApp) -> type[BaseHTTPRequestHandler]:
    class SVMRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stdout.write(f"[SVM-API] {self.address_string()} - {fmt % args}\n")

        def _get_cors_origin(self) -> str | None:
            origin = self.headers.get("Origin")
            if not origin:
                return None
            if origin in app.allowed_origins:
                return origin
            parsed = urlparse(origin)
            if parsed.scheme in {"http", "https"} and parsed.hostname in {
                "localhost",
                "127.0.0.1",
            }:
                return origin
            return None

        def _send_json(
            self, status_code: int, payload: dict[str, object] | list[object]
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            cors_origin = self._get_cors_origin()
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            cors_origin = self._get_cors_origin()
            if self.headers.get("Origin") and not cors_origin:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "Origin không nằm trong danh sách phát triển được phép (CORS)."},
                )
                return

            self.send_response(HTTPStatus.NO_CONTENT)
            if cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
                self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            parsed_path = urlparse(self.path).path.rstrip("/") or "/"
            if parsed_path in {"/health", "/api/health", "/"}:
                p = app.predictor
                lm_health_fn = getattr(app, "get_landmark_v2_health", None)
                lm_health = (
                    lm_health_fn()
                    if callable(lm_health_fn)
                    else {
                        "available": False,
                        "experimental": True,
                        "model_path": None,
                        "feature_schema": None,
                        "feature_dim": None,
                        "class_count": None,
                        "candidate_id": None,
                        "error": None,
                    }
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model_ready": True,
                        "model_path": str(app.model_path),
                        "model_type": type(p.classifier).__name__,
                        "c": float(p.bundle.get("c", getattr(p.classifier, "C", 0.01))),
                        "feature_count": p.n_features_in,
                        "class_count": len(p.ordered_class_labels),
                        "labels": p.ordered_class_labels,
                        "image_size": [64, 64],
                        "hog_parameters": {
                            "orientations": 9,
                            "pixels_per_cell": [8, 8],
                            "cells_per_block": [2, 2],
                            "block_norm": "L2-Hys",
                        },
                        "score_type": "decision_score",
                        "probability_calibrated": False,
                        "landmark_v2_available": bool(lm_health.get("available", False)),
                        "landmark_v2": lm_health,
                        "max_payload_bytes": app.max_payload_bytes,
                        "max_image_bytes": app.max_image_bytes,
                        "diagnostic_save_dir": str(app.diagnostic_save_dir),
                        "uptime_seconds": round(time.time() - app.startup_time, 2),
                    },
                )
                return

            if parsed_path in {"/diagnostic/next-instance", "/api/diagnostic/next-instance"}:
                from src.model.diagnostic_storage import suggest_next_gesture_instance_id
                qs = parse_qs(urlparse(self.path).query)
                pid = (qs.get("participant_id") or [""])[0].strip()
                sid = (qs.get("session_id") or [""])[0].strip()
                lbl = (qs.get("label") or [""])[0].strip().upper()
                info = suggest_next_gesture_instance_id(app.diagnostic_save_dir, pid, sid, lbl)
                info["diagnostic_save_dir"] = str(app.diagnostic_save_dir)
                self._send_json(HTTPStatus.OK, info)
                return

            if parsed_path in {"/labels", "/api/labels"}:
                self._send_json(HTTPStatus.OK, app.predictor.ordered_class_labels)
                return

            if parsed_path in {"/reference-samples", "/api/reference-samples"}:
                samples = app.get_reference_samples()
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "note": (
                            "Ảnh tham chiếu lấy từ tập train (asl_processed) cho các lớp A, B, C, J. "
                            "Đây là ảnh tham chiếu kỹ thuật, không phải tập đánh giá mới."
                        ),
                        "samples": samples,
                    },
                )
                return

            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"Không tìm thấy đường dẫn API: {parsed_path}"},
            )

        def _read_json_body(self) -> dict[str, object] | None:
            content_length_str = self.headers.get("Content-Length")
            if not content_length_str:
                self._send_json(
                    HTTPStatus.LENGTH_REQUIRED,
                    {"error": "Thiếu header Content-Length trong yêu cầu POST."},
                )
                return None

            try:
                content_length = int(content_length_str)
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Header Content-Length không hợp lệ."},
                )
                return None

            if content_length <= 0:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Nội dung yêu cầu (request body) rỗng."},
                )
                return None

            if content_length > app.max_payload_bytes:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {
                        "error": (
                            f"Kích thước dữ liệu gửi lên ({content_length} bytes) vượt quá "
                            f"giới hạn cho phép ({app.max_payload_bytes} bytes)."
                        )
                    },
                )
                return None

            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Dữ liệu gửi lên không phải là JSON hợp lệ."},
                )
                return None

            if not isinstance(payload, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "JSON gửi lên phải là một đối tượng (object)."},
                )
                return None

            return payload

        def do_POST(self) -> None:
            parsed_path = urlparse(self.path).path.rstrip("/")
            if parsed_path in {"/diagnostic/save", "/api/diagnostic/save"}:
                self._handle_diagnostic_save()
                return

            if parsed_path in {"/predict/landmark", "/api/predict/landmark"}:
                self._handle_predict_landmark()
                return

            if parsed_path not in {"/predict", "/api/predict"}:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Không tìm thấy endpoint POST: {parsed_path}"},
                )
                return

            payload = self._read_json_body()
            if payload is None:
                return

            image_b64 = payload.get("image")
            if not isinstance(image_b64, str) or not image_b64.strip():
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": (
                            "Thiếu trường 'image' (chuỗi base64 JPEG/PNG). "
                            "Mô hình SVM yêu cầu đầu vào là ảnh, không nhận mảng landmarks."
                        )
                    },
                )
                return

            cleaned_b64 = image_b64.strip()
            if "," in cleaned_b64 and cleaned_b64.lower().startswith("data:image"):
                cleaned_b64 = cleaned_b64.split(",", 1)[1].strip()

            try:
                image_bytes = base64.b64decode(cleaned_b64, validate=True)
            except (binascii.Error, ValueError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Chuỗi base64 trong trường 'image' không hợp lệ."},
                )
                return

            if len(image_bytes) == 0:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Dữ liệu ảnh sau khi giải mã base64 bị rỗng."},
                )
                return

            if len(image_bytes) > app.max_image_bytes:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {
                        "error": (
                            f"Kích thước ảnh giải mã ({len(image_bytes)} bytes) vượt quá "
                            f"giới hạn ({app.max_image_bytes} bytes)."
                        )
                    },
                )
                return

            top_k = payload.get("top_k", 3)
            try:
                top_k_int = int(top_k)  # type: ignore[arg-type]
                if top_k_int <= 0 or top_k_int > 36:
                    raise ValueError
            except (TypeError, ValueError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Tham số 'top_k' phải là số nguyên từ 1 đến 36."},
                )
                return

            crop_box = payload.get("crop_box")
            if crop_box is not None and not isinstance(crop_box, dict):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "Tham số 'crop_box' phải là một object {x, y, width, height} hoặc null."
                    },
                )
                return

            include_diagnostics = bool(payload.get("include_diagnostics", False))
            experimental_webcam_mode = bool(
                payload.get("experimental_webcam_mode", False)
                or payload.get("webcam_mode") == "method_3b"
            )
            already_flipped = bool(payload.get("already_flipped", False))
            frame_id = payload.get("frame_id")
            if frame_id is not None and not isinstance(frame_id, (str, int)):
                frame_id = str(frame_id)

            try:
                prediction_result = app.predictor.predict_image_bytes(
                    image_bytes=image_bytes,
                    top_k=top_k_int,
                    crop_box=crop_box,  # type: ignore[arg-type]
                    include_diagnostics=include_diagnostics,
                    frame_id=frame_id,
                    experimental_webcam_mode=experimental_webcam_mode,
                    already_flipped=already_flipped,
                )
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Lỗi nội bộ khi dự đoán SVM: {exc}"},
                )
                return

            self._send_json(HTTPStatus.OK, prediction_result)

        def _handle_predict_landmark(self) -> None:
            """Predict ASL sign from 21 MediaPipe landmarks across scope `all_36 | letters_az | digits_09`."""
            payload = self._read_json_body()
            if payload is None:
                return

            raw_scope = payload.get("scope")
            if raw_scope is None:
                scope = "all_36"
            elif isinstance(raw_scope, str) and raw_scope in VALID_LANDMARK_SCOPES:
                scope = raw_scope
            else:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": (
                            f"Tham số 'scope' không hợp lệ ({raw_scope!r}). "
                            f"Chỉ chấp nhận: {', '.join(VALID_LANDMARK_SCOPES)}."
                        )
                    },
                )
                return

            registry = getattr(app, "landmark_registry", None)
            if registry is not None:
                lm_predictor = registry.get_predictor(scope)
            elif scope == "all_36":
                lm_predictor = getattr(app, "landmark_v2_predictor", None)
            else:
                lm_predictor = None

            if lm_predictor is None:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": (
                            f"Model Landmark SVM cho phạm vi '{scope}' hiện không sẵn sàng trên máy chủ."
                        )
                    },
                )
                return

            top_k = payload.get("top_k", 3)
            try:
                if isinstance(top_k, bool):
                    raise ValueError
                top_k_int = int(top_k)  # type: ignore[arg-type]
                if top_k_int <= 0 or top_k_int > 36:
                    raise ValueError
            except (TypeError, ValueError):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Tham số 'top_k' phải là số nguyên từ 1 đến 36."},
                )
                return

            frame_id = payload.get("frame_id")
            if frame_id is not None and not isinstance(frame_id, (str, int)):
                frame_id = str(frame_id)

            try:
                result = lm_predictor.predict_from_landmarks(
                    landmarks=payload.get("landmarks"),
                    image_width=payload.get("image_width"),
                    image_height=payload.get("image_height"),
                    top_k=top_k_int,
                    frame_id=frame_id,
                )
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Lỗi nội bộ khi dự đoán Landmark SVM ({scope}): {exc}"},
                )
                return

            self._send_json(HTTPStatus.OK, result)

        def _handle_diagnostic_save(self) -> None:
            """Save a single diagnostic snapshot when explicitly triggered by the user."""
            from src.model.diagnostic_storage import save_snapshot
            body = self._read_json_body()
            if body is None:
                return
            try:
                result = save_snapshot(body, app.diagnostic_save_dir)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {'error': str(exc)})
            except OSError as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {'error': 'Snapshot write failed: ' + str(exc)})
            else:
                self._send_json(HTTPStatus.OK, result)

    return SVMRequestHandler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Khởi chạy máy chủ API Python nhận diện ký hiệu ASL (Landmark SVM 3 phạm vi + HOG baseline)."
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Địa chỉ host lắng nghe (mặc định: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Cổng lắng nghe (mặc định: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Đường dẫn tới file model HOG baseline (mặc định: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--landmark-v2-model",
        type=Path,
        default=DEFAULT_LANDMARK_V2_PATH_CONFIG,
        help=f"Đường dẫn model Landmark SVM 36 lớp (mặc định: {DEFAULT_LANDMARK_V2_PATH_CONFIG})",
    )
    parser.add_argument(
        "--landmark-az-model",
        type=Path,
        default=DEFAULT_LANDMARK_AZ_PATH_CONFIG,
        help=f"Đường dẫn model Landmark SVM A–Z 26 lớp (mặc định: {DEFAULT_LANDMARK_AZ_PATH_CONFIG})",
    )
    parser.add_argument(
        "--landmark-09-model",
        type=Path,
        default=DEFAULT_LANDMARK_09_PATH_CONFIG,
        help=f"Đường dẫn model Landmark SVM 0–9 10 lớp (mặc định: {DEFAULT_LANDMARK_09_PATH_CONFIG})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_DIAGNOSTIC_SAVE_DIR,
        help=(
            "Thư mục lưu mẫu khi bấm 'Lưu mẫu chẩn đoán' "
            f"(mặc định: {DEFAULT_DIAGNOSTIC_SAVE_DIR}, có thể đặt qua biến môi trường SVM_DIAGNOSTIC_SAVE_DIR)"
        ),
    )
    parser.add_argument(
        "--max-payload-bytes",
        type=int,
        default=DEFAULT_MAX_PAYLOAD_BYTES,
        help="Giới hạn kích thước JSON gửi lên (mặc định: 4MB)",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=DEFAULT_MAX_IMAGE_BYTES,
        help="Giới hạn kích thước byte ảnh sau giải mã base64 (mặc định: 3MB)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args(argv)
    print(f"[SVM-API] Đang nạp model baseline HOG từ: {args.model.resolve()} ...")
    app = SVMAPIApp(
        model_path=args.model,
        allowed_origins=DEFAULT_ALLOWED_ORIGINS,
        max_payload_bytes=args.max_payload_bytes,
        max_image_bytes=args.max_image_bytes,
        diagnostic_save_dir=args.save_dir,
        landmark_v2_model_path=args.landmark_v2_model,
        landmark_az_model_path=args.landmark_az_model,
        landmark_09_model_path=args.landmark_09_model,
    )
    print(
        f"[SVM-API] Đã nạp xong HOG baseline (C={app.predictor.bundle.get('c', 0.01)}, "
        f"{app.predictor.n_features_in} đặc trưng HOG, {len(app.predictor.ordered_class_labels)} lớp)."
    )
    for sc in VALID_LANDMARK_SCOPES:
        sh = app.landmark_registry.get_scope_health(sc)
        if sh["available"]:
            print(
                f"[SVM-API] Landmark SVM [{sc:10s}] SẴN SÀNG: {sh['class_count']} lớp | "
                f"{LANDMARK_GEOM_V2_FEATURE_SCHEMA} ({LANDMARK_GEOM_V2_FEATURE_DIM} đặc trưng)."
            )
        else:
            print(f"[SVM-API] Landmark SVM [{sc:10s}] KHÔNG KHẢ DỤNG: {sh['error']}")

    handler_cls = create_request_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"[SVM-API] Máy chủ đang chạy tại: http://{args.host}:{args.port}")
    print(f"          - Kiểm tra trạng thái     : http://{args.host}:{args.port}/health")
    print(f"          - Dự đoán Landmark (POST) : http://{args.host}:{args.port}/predict/landmark")
    print(f"          - Dự đoán HOG cũ (POST)   : http://{args.host}:{args.port}/predict")
    print(f"          - Lưu mẫu chẩn đoán       : http://{args.host}:{args.port}/diagnostic/save")
    print(f"          - Thư mục lưu mẫu         : {app.diagnostic_save_dir.resolve()}")
    print("          Nhấn Ctrl+C để dừng máy chủ.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SVM-API] Đang đóng máy chủ...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
