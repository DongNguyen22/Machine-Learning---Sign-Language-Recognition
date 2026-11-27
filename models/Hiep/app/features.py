"""Trích xuất vector 181 đặc trưng từ MediaPipe HandLandmarkerResult."""
from __future__ import annotations

import numpy as np

# Các bộ 3 điểm để tính 15 góc gập ngón tay
ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # Ngón cái
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # Ngón trỏ
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # Ngón giữa
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # Ngón áp út
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # Ngón út
]
FINGERTIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(FINGERTIPS[i], FINGERTIPS[j]) for i in range(5) for j in range(i + 1, 5)]


def extract_181_features(result, w: int, h: int) -> tuple[np.ndarray, bool]:
    """Trả về (vector 181 chiều, có phát hiện tay hay không)."""
    hands = np.zeros((2, 21, 3), dtype=np.float32)
    mask = np.zeros(2, dtype=np.float32)
    if not result.hand_landmarks:
        return np.zeros(181, dtype=np.float32), False

    for i, lms in enumerate(result.hand_landmarks[:2]):
        hands[i] = [(p.x * w, p.y * h, p.z * w) for p in lms]
        mask[i] = 1.0

    # Sắp xếp tay có mặt lên trước, từ trái sang phải theo cổ tay
    key = np.where(mask > 0, hands[:, 0, 0], np.inf)
    order = np.argsort(key, kind="stable")
    hands, mask = hands[order], mask[order]

    feats = []
    scales = []
    wrists = []
    for i in range(2):
        if mask[i] == 0:
            feats.append(np.zeros(63 + 15 + 10, dtype=np.float32))
            scales.append(1.0)
            wrists.append(np.zeros(3, dtype=np.float32))
            continue

        wrist = hands[i, 0].copy()
        rel = hands[i] - wrist
        scale = max(float(np.linalg.norm(rel, axis=-1).max()), 1e-6)
        norm_pts = rel / scale

        # 15 góc gập ngón (radian)
        angles = np.zeros(15, dtype=np.float32)
        for idx, (a, b, c) in enumerate(ANGLE_TRIPLETS):
            v1 = norm_pts[a] - norm_pts[b]
            v2 = norm_pts[c] - norm_pts[b]
            denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
            cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            angles[idx] = np.arccos(cos_val)

        # 10 khoảng cách đầu ngón
        dists = np.array(
            [np.linalg.norm(norm_pts[p] - norm_pts[q]) for p, q in TIP_PAIRS],
            dtype=np.float32,
        )

        feats.append(np.concatenate([norm_pts.reshape(-1), angles, dists]))
        scales.append(scale)
        wrists.append(wrist)

    # Vector giữa 2 cổ tay nếu có 2 tay
    if mask[0] > 0 and mask[1] > 0:
        wrist_vec = (wrists[1] - wrists[0]) / scales[0]
    else:
        wrist_vec = np.zeros(3, dtype=np.float32)

    return np.concatenate([feats[0], feats[1], wrist_vec, mask]).astype(np.float32), True
