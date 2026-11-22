"""Đặc trưng landmark bàn tay dùng chung cho TRAIN (notebook) và DEPLOY (FastAPI).

Hai bên phải dùng đúng cùng một file này. Nếu sửa cách chuẩn hoá, tăng FEATURE_VERSION
và train lại; app sẽ từ chối model có feature_version khác.

Biểu diễn 1 mẫu:
    hands: (2, 21, 3) float32, toạ độ pixel (x*W, y*H, z*W) của tối đa 2 bàn tay
    mask : (2,) float32, 1 = có tay ở ô đó
    Tay có mặt luôn xếp trước; nếu có 2 tay thì xếp theo cổ tay từ trái sang phải trong ảnh.

Vector đặc trưng (131 số):
    [tay 1: 63] [tay 2: 63] [vector cổ tay 1 -> cổ tay 2: 3] [mask: 2]
    Mỗi tay: trừ toạ độ cổ tay (bất biến vị trí), chia cho kích thước tay (bất biến khoảng cách).
    KHÔNG chuẩn hoá xoay, vì nhiều cặp chữ chỉ khác nhau ở hướng tay (H/U, K/P, G/Q).
"""
from __future__ import annotations

import io
from typing import BinaryIO, Sequence

import numpy as np
from PIL import Image, ImageOps

FEATURE_VERSION = 1
MAX_SIDE = 640          # cạnh dài tối đa của ảnh trước khi đưa vào MediaPipe
NUM_HANDS = 2           # số 0 trong ASL-HG là ký hiệu 2 tay
N_POINTS = 21
FEATURE_DIM = NUM_HANDS * N_POINTS * 3 + 3 + NUM_HANDS   # 131
_EPS = 1e-6


# ---------------------------------------------------------------- đọc ảnh
def load_image_rgb(src: str | bytes | BinaryIO, max_side: int = MAX_SIDE) -> np.ndarray:
    """Đọc ảnh -> mảng RGB uint8 (H, W, 3). Xoay theo EXIF (ảnh điện thoại hay bị xoay)."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    img = Image.open(src)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


# ---------------------------------------------------------------- landmark -> mảng
def sort_hands(hands: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Xếp tay có mặt lên trước, theo x cổ tay tăng dần. Nhận (2,21,3)/(2,) hoặc (N,2,21,3)/(N,2)."""
    single = hands.ndim == 3
    if single:
        hands, mask = hands[None], mask[None]
    key = np.where(mask > 0, hands[:, :, 0, 0], np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    hands = np.take_along_axis(hands, order[:, :, None, None], axis=1)
    mask = np.take_along_axis(mask, order, axis=1)
    return (hands[0], mask[0]) if single else (hands, mask)


def result_to_hands(result, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """mediapipe HandLandmarkerResult -> (hands (2,21,3) pixel, mask (2,))."""
    hands = np.zeros((NUM_HANDS, N_POINTS, 3), np.float32)
    mask = np.zeros(NUM_HANDS, np.float32)
    for i, lms in enumerate(result.hand_landmarks[:NUM_HANDS]):
        hands[i] = [(p.x * width, p.y * height, p.z * width) for p in lms]
        mask[i] = 1.0
    return sort_hands(hands, mask)


def normalized_to_hands(hand_list: Sequence[Sequence[Sequence[float]]],
                        width: float = 1.0, height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Danh sách tay [[[x,y,z] x21], ...] với x,y trong [0,1] (như MediaPipe JS trả về) -> mảng."""
    hands = np.zeros((NUM_HANDS, N_POINTS, 3), np.float32)
    mask = np.zeros(NUM_HANDS, np.float32)
    for i, h in enumerate(list(hand_list)[:NUM_HANDS]):
        a = np.asarray(h, dtype=np.float32)
        if a.shape != (N_POINTS, 3):
            raise ValueError(f"Mỗi tay phải có shape (21, 3), nhận {a.shape}")
        hands[i] = a * np.array([width, height, width], np.float32)
        mask[i] = 1.0
    return sort_hands(hands, mask)


# ---------------------------------------------------------------- đặc trưng
def featurize_batch(hands: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(N,2,21,3), (N,2) -> (N, FEATURE_DIM) float32."""
    hands = hands.astype(np.float32)
    mask = mask.astype(np.float32)
    wrist = hands[:, :, :1, :]                                  # (N,2,1,3)
    rel = hands - wrist
    scale = np.linalg.norm(rel, axis=-1).max(axis=-1)           # (N,2) khoảng cách xa nhất từ cổ tay
    scale = np.maximum(scale, _EPS)
    rel = rel / scale[:, :, None, None] * mask[:, :, None, None]

    both = mask[:, 0] * mask[:, 1]
    mean_scale = np.where(both > 0, scale.mean(axis=1), 1.0)
    inter = (hands[:, 1, 0] - hands[:, 0, 0]) / mean_scale[:, None] * both[:, None]

    n = hands.shape[0]
    return np.concatenate([rel.reshape(n, -1), inter, mask], axis=1).astype(np.float32)


def featurize(hands: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(2,21,3), (2,) -> (FEATURE_DIM,)"""
    return featurize_batch(hands[None], mask[None])[0]


# ---------------------------------------------------------------- augmentation (chỉ dùng khi train)
def augment_batch(hands: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                  rot_deg: float = 15.0, flip_p: float = 0.5, jitter: float = 0.015,
                  stretch: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Biến đổi hình học trên toạ độ (rẻ hơn rất nhiều so với augment ảnh).

    - Lật ngang (người thuận tay trái ký hiệu đối xứng)
    - Xoay nhẹ trong mặt phẳng ảnh (giữ nhỏ: xoay lớn biến H thành U, K thành P)
    - Kéo giãn x/y khác nhau (góc chụp, tỉ lệ khung hình)
    - Nhiễu nhỏ trên từng điểm (sai số của MediaPipe)
    """
    n = hands.shape[0]
    h = hands.astype(np.float32).copy()

    flip = rng.random(n) < flip_p
    h[flip, :, :, 0] *= -1.0

    th = np.deg2rad(rng.uniform(-rot_deg, rot_deg, n)).astype(np.float32)
    c, s = np.cos(th), np.sin(th)
    x, y = h[..., 0].copy(), h[..., 1].copy()
    h[..., 0] = c[:, None, None] * x - s[:, None, None] * y
    h[..., 1] = s[:, None, None] * x + c[:, None, None] * y

    sx = rng.uniform(1 - stretch, 1 + stretch, n).astype(np.float32)
    sy = rng.uniform(1 - stretch, 1 + stretch, n).astype(np.float32)
    h[..., 0] *= sx[:, None, None]
    h[..., 1] *= sy[:, None, None]

    size = np.linalg.norm(h - h[:, :, :1], axis=-1).max(axis=-1)          # (N,2)
    h += rng.normal(0, 1, h.shape).astype(np.float32) * (jitter * size)[:, :, None, None]
    h *= mask[:, :, None, None]

    return sort_hands(h, mask.astype(np.float32).copy())
