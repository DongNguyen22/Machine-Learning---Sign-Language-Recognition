"""API nhận diện ký hiệu ASL (A-Z, 0-9) từ ảnh, dùng MediaPipe Hands + MLP trên landmark.

Chạy:  uvicorn app.main:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs      Demo webcam: http://localhost:8000/
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .classifier import MLPClassifier
from .features import MAX_SIDE, featurize, load_image_rgb, normalized_to_hands

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", HERE / "model"))
DETECTOR_POOL = int(os.getenv("DETECTOR_POOL", "2"))
MIN_HAND_CONF = float(os.getenv("MIN_HAND_CONF", "0.3"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = MLPClassifier(MODEL_DIR / "asl_mlp.npz", MODEL_DIR / "model_meta.json")
    if "detector" not in state:             # test có thể gắn detector giả trước
        from .landmarker import HandDetectorPool, ensure_model
        task = ensure_model(MODEL_DIR / "hand_landmarker.task")
        state["detector"] = HandDetectorPool(task, size=DETECTOR_POOL, min_conf=MIN_HAND_CONF)
    yield
    det = state.pop("detector", None)
    if det is not None and hasattr(det, "close"):
        det.close()


app = FastAPI(title="ASL-HG Sign Recognition", version="1.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

if os.getenv("ENABLE_COLLECT", "1") == "1":          # trang ghi mẫu /collect; đặt ENABLE_COLLECT=0 khi deploy công khai
    from .collect import make_router
    app.include_router(make_router(state))


# ------------------------------------------------------------------ schema
class Candidate(BaseModel):
    label: str
    prob: float


class Prediction(BaseModel):
    hand_detected: bool
    num_hands: int
    label: str | None = None
    confidence: float | None = None
    accepted: bool = Field(False, description=f"confidence >= ngưỡng (CONF_THRESHOLD)")
    top_k: list[Candidate] = []
    landmarks: list[list[list[float]]] | None = Field(
        None, description="Toạ độ chuẩn hoá [0,1] của từng tay, để vẽ lên ảnh")
    latency_ms: float


class LandmarkRequest(BaseModel):
    hands: list[list[list[float]]] = Field(
        ..., description="1-2 tay, mỗi tay 21 điểm [x, y, z] chuẩn hoá như MediaPipe trả về")
    image_width: float = Field(1.0, gt=0, description="Chiều rộng ảnh gốc (để giữ đúng tỉ lệ)")
    image_height: float = Field(1.0, gt=0)


# ------------------------------------------------------------------ logic
Mode = Literal["all", "letters", "digits"]


def _predict(hands: np.ndarray, mask: np.ndarray, top_k: int, t0: float,
             wh: tuple[float, float] | None, return_landmarks: bool, mode: str = "all") -> Prediction:
    n = int(mask.sum())
    lms = None
    if return_landmarks and wh and n:
        w, h = wh
        lms = (hands[:n] / np.array([w, h, w], np.float32)).round(4).tolist()
    if n == 0:
        return Prediction(hand_detected=False, num_hands=0, landmarks=lms,
                          latency_ms=(time.perf_counter() - t0) * 1000)
    cands = state["clf"].top_k(featurize(hands, mask), k=top_k, mode=mode)
    best = cands[0]
    return Prediction(hand_detected=True, num_hands=n, label=best["label"], confidence=best["prob"],
                      accepted=best["prob"] >= CONF_THRESHOLD, top_k=cands, landmarks=lms,
                      latency_ms=(time.perf_counter() - t0) * 1000)


# ------------------------------------------------------------------ routes
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
def health():
    clf = state.get("clf")
    return {"status": "ok" if clf and state.get("detector") else "loading",
            "num_classes": len(clf.labels) if clf else 0,
            "feature_version": clf.meta.get("feature_version") if clf else None,
            "test_accuracy": clf.meta.get("test_accuracy") if clf else None}


@app.get("/labels")
def labels():
    return state["clf"].labels


@app.post("/predict", response_model=Prediction)
def predict_image(file: UploadFile = File(..., description="Ảnh JPG/PNG có bàn tay"),
                  top_k: int = Query(5, ge=1, le=36),
                  return_landmarks: bool = Query(True),
                  mode: Mode = Query("all", description="all = 36 lớp, letters = chỉ A-Z, digits = chỉ 0-9")):
    """Gửi 1 ảnh -> MediaPipe tìm tay trên server -> phân loại."""
    t0 = time.perf_counter()
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Ảnh lớn hơn {MAX_UPLOAD_MB} MB")
    try:
        rgb = load_image_rgb(data, MAX_SIDE)
    except Exception:
        raise HTTPException(400, "Không đọc được ảnh (cần JPG/PNG hợp lệ)")
    hands, mask = state["detector"].detect(rgb)
    return _predict(hands, mask, top_k, t0, (rgb.shape[1], rgb.shape[0]), return_landmarks, mode)


@app.post("/predict/landmarks", response_model=Prediction)
def predict_landmarks(req: LandmarkRequest, top_k: int = Query(5, ge=1, le=36),
                      mode: Mode = Query("all", description="all = 36 lớp, letters = chỉ A-Z, digits = chỉ 0-9")):
    """Client tự chạy MediaPipe (JS/Android/iOS) và chỉ gửi 21 điểm/tay: nhanh, tiết kiệm băng thông."""
    t0 = time.perf_counter()
    if not 1 <= len(req.hands) <= 2:
        raise HTTPException(422, "Cần 1 hoặc 2 tay")
    try:
        hands, mask = normalized_to_hands(req.hands, req.image_width, req.image_height)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return _predict(hands, mask, top_k, t0, None, False, mode)
