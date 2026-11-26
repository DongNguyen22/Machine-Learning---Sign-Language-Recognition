"""Rút vector hình học 91 chiều từ landmark đã chuẩn hóa."""

from __future__ import annotations

import numpy as np

FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
EPSILON = 1e-6


def _safe_angle(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= EPSILON or norm_b <= EPSILON:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def calculate_joint_angle(
    left_point: np.ndarray,
    vertex: np.ndarray,
    right_point: np.ndarray,
) -> float:
    """Tính góc gập tại khớp bằng arccos tích vô hướng hai vector đốt ngón."""
    left = np.asarray(left_point, dtype=np.float32)
    joint = np.asarray(vertex, dtype=np.float32)
    right = np.asarray(right_point, dtype=np.float32)
    if left.shape != (3,) or joint.shape != (3,) or right.shape != (3,):
        raise ValueError("Các điểm phải là vector 3 chiều.")
    if not (
        np.isfinite(left).all()
        and np.isfinite(joint).all()
        and np.isfinite(right).all()
    ):
        raise ValueError("Các điểm không được chứa NaN hoặc Inf.")
    return _safe_angle(left - joint, right - joint)


def extract_15_finger_angles(points: np.ndarray) -> np.ndarray:
    """Lấy ba góc gập liên tiếp trên mỗi ngón, gồm 15 góc."""
    angles: list[float] = []
    for chain in FINGER_CHAINS:
        a, b, c, d = chain
        for vertex, left, right in ((b, a, c), (c, b, d), (d, c, d)):
            vector_a = points[left] - points[vertex]
            vector_b = points[right] - points[vertex]
            angles.append(_safe_angle(vector_a, vector_b))
    if len(angles) != 15:
        raise RuntimeError("Không tạo đủ 15 góc ngón tay.")
    return np.asarray(angles, dtype=np.float32)


def extract_geometric_features(
    normalized_landmarks: np.ndarray,
    baseline_63: bool = False,
    return_baseline_63: bool | None = None,
) -> np.ndarray:
    """Ghép 63 tọa độ, 15 góc gập, 10 khoảng cách đầu ngón và 3 thành phần pháp tuyến."""
    points = np.asarray(normalized_landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError("Landmarks chuẩn hóa phải có shape (21, 3).")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks không được chứa NaN hoặc Inf.")

    if return_baseline_63 is not None:
        baseline_63 = return_baseline_63

    coordinates = points.reshape(63).astype(np.float32)
    if baseline_63:
        result = coordinates
    else:
        angles: list[float] = []
        for chain in FINGER_CHAINS:
            a, b, c, d = chain
            for vertex, left, right in ((b, a, c), (c, b, d), (d, c, d)):
                vector_a = points[left] - points[vertex]
                vector_b = points[right] - points[vertex]
                angles.append(_safe_angle(vector_a, vector_b))
        angles_array = np.asarray(angles, dtype=np.float32)

        tip_indexes = np.asarray([4, 8, 12, 16, 20], dtype=np.int32)
        wrist_tip = np.linalg.norm(points[tip_indexes] - points[0], axis=1)
        adjacent_pairs = np.asarray(
            [(4, 8), (8, 12), (12, 16), (16, 20), (4, 20)],
            dtype=np.int32,
        )
        tip_distances = np.linalg.norm(
            points[adjacent_pairs[:, 0]] - points[adjacent_pairs[:, 1]],
            axis=1,
        )
        distances = np.concatenate((wrist_tip, tip_distances)).astype(np.float32)

        palm_a = points[5] - points[0]
        palm_b = points[17] - points[0]
        normal = np.cross(palm_a, palm_b)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > EPSILON:
            normal = normal / normal_norm
        else:
            normal = np.zeros(3, dtype=np.float32)
        normal = normal.astype(np.float32)

        result = np.concatenate((coordinates, angles_array, distances, normal)).astype(np.float32)

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result = np.asarray(result, dtype=np.float32)
    return result


extract_91_features = extract_geometric_features
extract_features = extract_geometric_features


class FeatureExtractor:
    """Tạo vector hình học 63 hoặc 91 chiều."""

    def transform(
        self,
        landmarks: np.ndarray,
        baseline_63: bool = False,
    ) -> np.ndarray:
        return extract_geometric_features(
            landmarks,
            baseline_63=baseline_63,
        )
