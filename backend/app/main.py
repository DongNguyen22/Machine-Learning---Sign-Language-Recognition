"""FastAPI backend nhận diện ASL A-Z cho UI trong thư mục frontend/.

Model: models/Triet/models/production/exp3_mlp_opt.onnx (MLP 91 đặc trưng, tốt nhất trong benchmark).

Chạy (từ thư mục backend/):
    uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------------------------------------------ cấu hình
REPO_ROOT = Path(__file__).resolve().parents[2]
TRIET_DIR = Path(os.getenv("TRIET_DIR", REPO_ROOT / "models" / "Triet"))
MODEL_NAME = os.getenv("MODEL_NAME", "exp3_mlp_opt")
MODEL_PATH = Path(os.getenv("MODEL_PATH", TRIET_DIR / "models" / "production" / f"{MODEL_NAME}.onnx"))
META_PATH = Path(os.getenv("META_PATH", MODEL_PATH.with_name(f"{MODEL_NAME}_metadata.json")))
# Số frame cuối trong chuỗi frontend gửi lên được dùng để lấy trung bình xác suất (làm mượt).
SMOOTH_FRAMES = int(os.getenv("SMOOTH_FRAMES", "5"))
TOP_K = int(os.getenv("TOP_K", "3"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8443,http://127.0.0.1:8443").split(",")


def _load_triet_module(name: str, relative_path: str):
    """Nạp file .py của Triết theo đường dẫn, KHÔNG thêm models/Triet vào sys.path
    (models/Triet cũng có package `app` -> sẽ đè lên package `app` của backend)."""
    spec = importlib.util.spec_from_file_location(name, TRIET_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Dùng lại đúng code tiền xử lý lúc train của Triết -> train/serve nhất quán.
extract_geometric_features = _load_triet_module(
    "triet_geometric_features", "src/feature_extraction/geometric_features.py"
).extract_geometric_features
normalize_landmarks = _load_triet_module(
    "triet_coordinate_normalizer", "src/preprocessing/coordinate_normalizer.py"
).normalize_landmarks

state: dict = {}


def load_labels(meta_path: Path) -> list[str]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_to_index: dict[str, int] = meta["label_to_index"]
    return [label for label, _ in sorted(label_to_index.items(), key=lambda kv: kv[1])]


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    state["session"] = session
    state["input_name"] = session.get_inputs()[0].name
    state["input_dim"] = session.get_inputs()[0].shape[1]
    state["labels"] = load_labels(META_PATH)
    yield
    state.clear()


app = FastAPI(title="ASL A-Z Recognition API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ------------------------------------------------------------------ schema (khớp frontend)
class PredictRequest(BaseModel):
    landmarks: list[list[float]] = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Chuỗi frame, mỗi frame 63 số [x0,y0,z0,...,x20,y20,z20] từ MediaPipe (toạ độ chuẩn hoá theo ảnh).",
    )


class Candidate(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_k: list[Candidate]
    latency_ms: float


# ------------------------------------------------------------------ logic
def frames_to_features(frames: list[list[float]]) -> np.ndarray:
    rows = []
    for i, frame in enumerate(frames):
        if len(frame) != 63:
            raise HTTPException(422, f"Frame {i} có {len(frame)} giá trị, cần đúng 63.")
        points = np.asarray(frame, dtype=np.float32).reshape(21, 3)
        if not np.isfinite(points).all():
            raise HTTPException(422, f"Frame {i} chứa NaN/Inf.")
        rows.append(extract_geometric_features(normalize_landmarks(points)))
    return np.stack(rows).astype(np.float32)


def predict_proba(features: np.ndarray) -> np.ndarray:
    return state["session"].run(None, {state["input_name"]: features})[0]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "num_classes": len(state["labels"])}


@app.get("/labels")
def labels() -> list[str]:
    return state["labels"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    started = time.perf_counter()
    frames = req.landmarks[-SMOOTH_FRAMES:]
    probs = predict_proba(frames_to_features(frames)).mean(axis=0)

    labels_ = state["labels"]
    order = np.argsort(probs)[::-1][:TOP_K]
    top_k = [Candidate(label=labels_[i], confidence=float(probs[i])) for i in order]
    return PredictResponse(
        label=top_k[0].label,
        confidence=top_k[0].confidence,
        top_k=top_k,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
