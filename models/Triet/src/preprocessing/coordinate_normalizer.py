"""Chuẩn hóa tọa độ landmark MediaPipe."""

from __future__ import annotations

import numpy as np


def normalize_landmarks(
    landmarks: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Cố định cổ tay làm gốc và chia theo d_max để vị trí bàn tay không phụ thuộc vị trí, tỷ lệ và góc nhìn."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError("Landmarks phải có shape (21, 3).")
    if epsilon <= 0:
        raise ValueError("epsilon phải dương.")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks không được chứa NaN hoặc Inf.")

    centered = points - points[0]
    distances = np.linalg.norm(centered, axis=1)
    scale = max(float(np.max(distances)), epsilon)
    normalized = centered / scale
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized.astype(np.float32), -1.0, 1.0)


normalize_coordinates = normalize_landmarks


class Normalizer:
    """Đặt cổ tay làm gốc và chuẩn hóa tỷ lệ bàn tay."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def transform(self, landmarks: np.ndarray) -> np.ndarray:
        return normalize_landmarks(landmarks, epsilon=self.epsilon)
