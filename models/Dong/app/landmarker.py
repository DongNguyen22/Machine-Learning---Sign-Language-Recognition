"""Bọc MediaPipe HandLandmarker (Tasks API). Dùng chung cho trích landmark khi train và cho API.

Lưu ý: mediapipe >= 0.10.30 đã bỏ `mp.solutions`, chỉ còn Tasks API và cần file model .task.
"""
from __future__ import annotations

import queue
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .features import NUM_HANDS, result_to_hands

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def ensure_model(path: str | Path) -> Path:
    """Tải hand_landmarker.task (~7.5 MB) nếu chưa có."""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1_000_000:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        if tmp.stat().st_size < 1_000_000:
            raise RuntimeError(f"Tải model thất bại (file {tmp.stat().st_size} byte). Tải tay từ {MODEL_URL}")
        tmp.replace(path)
    return path


def create_landmarker(model_path: str | Path, num_hands: int = NUM_HANDS, min_conf: float = 0.3):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detect_hands(landmarker, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ảnh RGB uint8 (H,W,3) -> (hands (2,21,3) toạ độ pixel, mask (2,))."""
    import mediapipe as mp

    h, w = rgb.shape[:2]
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return result_to_hands(result, w, h)


class HandDetectorPool:
    """Một HandLandmarker không an toàn khi gọi song song -> giữ một pool, mỗi request mượn một cái."""

    def __init__(self, model_path: str | Path, size: int = 2, min_conf: float = 0.3):
        self._q: queue.Queue = queue.Queue()
        self._all = []
        for _ in range(max(1, size)):
            lm = create_landmarker(model_path, min_conf=min_conf)
            self._all.append(lm)
            self._q.put(lm)

    @contextmanager
    def acquire(self, timeout: float = 30.0):
        lm = self._q.get(timeout=timeout)
        try:
            yield lm
        finally:
            self._q.put(lm)

    def detect(self, rgb: np.ndarray):
        with self.acquire() as lm:
            return detect_hands(lm, rgb)

    def close(self):
        for lm in self._all:
            try:
                lm.close()
            except Exception:
                pass
