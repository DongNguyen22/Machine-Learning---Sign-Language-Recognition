"""FastAPI server + Web UI cho mô hình SA-ResMLP (Hiệp).

Chạy: uvicorn app.main:app --host 0.0.0.0 --port 8000
Mở trình duyệt: http://localhost:8000/
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .features import extract_181_features
from .predict import load_predictor

HERE = Path(__file__).resolve().parent
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, labels, mean, std, device, landmarker = load_predictor()
    state.update(model=model, labels=labels, mean=mean, std=std, device=device, landmarker=landmarker)
    yield
    lm = state.pop("landmarker", None)
    if lm is not None:
        lm.close()


app = FastAPI(title="SA-ResMLP ASL Recognition (Hiep)", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=36),
    mode: Literal["all", "letters", "digits"] = Query("all"),
):
    import mediapipe as mp
    import torch

    t0 = time.perf_counter()
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh tải lên.")

    h, w = bgr.shape[:2]
    if max(h, w) > 640:
        s = 640 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        h, w = bgr.shape[:2]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = state["landmarker"].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    vec, ok = extract_181_features(res, w, h)
    if not ok:
        return {
            "hand_detected": False,
            "num_hands": 0,
            "label": None,
            "confidence": None,
            "top_k": [],
            "landmarks": [],
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    x = (vec[None, :] - state["mean"]) / state["std"]
    with torch.no_grad():
        probs = torch.softmax(
            state["model"](torch.from_numpy(x).to(state["device"])), dim=-1
        )[0].cpu().numpy()

    labels = state["labels"]
    if mode != "all":
        keep = np.array([lab.isdigit() == (mode == "digits") for lab in labels])
        probs = np.where(keep, probs, 0.0)
        probs = probs / max(float(probs.sum()), 1e-12)

    idx = [i for i in np.argsort(probs)[::-1] if probs[i] > 0][:top_k]
    top = [{"label": labels[i], "prob": float(probs[i])} for i in idx]
    lms = [[[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in hand] for hand in res.hand_landmarks]

    return {
        "hand_detected": True,
        "num_hands": len(res.hand_landmarks),
        "label": top[0]["label"],
        "confidence": top[0]["prob"],
        "top_k": top,
        "landmarks": lms,
        "latency_ms": (time.perf_counter() - t0) * 1000,
    }
