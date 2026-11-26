"""Shared feature extraction for SignVision webcam training & inference parity.

Strictly separates two distinct feature pipelines:
1. HOG (1764 float32 features):
   - Reuses `apply_webcam_method_3b_from_bgr` and `extract_hog_with_diagnostics_from_bgr`
     from `src/preprocess/build_hog_dataset.py` so training features match live webcam
     inference (`SVMSignPredictor.predict_image_bgr`) bit-for-bit.
   - Canonicalizes saved crops (`received_crop.png`) using saved orientation metadata so
     no sample is accidentally flipped twice.

2. MediaPipe 21 Hand Landmarks (42 float32 features — SEPARATE EXPERIMENT):
   - Inspired by `computervisioneng/sign-language-detector-python` (`create_dataset.py`),
     adapted cleanly for SVM (`LinearSVC` / `SVC`), NOT Random Forest.
   - Extracts 21 `(x, y)` hand landmarks via `mediapipe.tasks.python.vision.HandLandmarker`,
     translates relative to wrist landmark 0 `(x_i - x_0, y_i - y_0)`, and normalizes by
     the hand's maximum Euclidean radius from the wrist so features are invariant to
     hand translation and camera distance (scale).
   - Never feeds 42-dim landmarks into any 1764-dim HOG model.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_hog_dataset import (
    apply_webcam_method_3b_from_bgr,
    extract_hog_with_diagnostics_from_bgr,
)

HOG_FEATURE_SCHEMA = "hog_64x64_9ori_8ppc_2cpb_l2hys_1764_v1"
HOG_FEATURE_DIM = 1764

LANDMARK_FEATURE_SCHEMA = "mediapipe_21_xy_wrist_norm_aspect_v2"
LANDMARK_FEATURE_DIM = 42
LANDMARK_GEOM_V2_FEATURE_SCHEMA = "mediapipe_21_xy_geom_v2"
LANDMARK_GEOM_V2_FEATURE_DIM = 86
DEFAULT_HAND_LANDMARKER_MODEL = PROJECT_ROOT / "results" / "svm" / "hand_landmarker.task"

_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
_FINGERTIP_PAIRS = (
    (4, 8),
    (4, 12),
    (4, 16),
    (4, 20),
    (8, 12),
    (8, 16),
    (8, 20),
    (12, 16),
    (12, 20),
    (16, 20),
)
_TIP_TO_MCP_PAIRS = ((4, 2), (8, 5), (12, 9), (16, 13), (20, 17))
_THUMB_TO_KNUCKLE_PAIRS = ((4, 5), (4, 9), (4, 13), (4, 17), (4, 10))
_JOINT_ANGLE_TRIPLETS = (
    # Thumb: CMC, MCP, IP
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    # Index: MCP, PIP, DIP
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    # Middle: MCP, PIP, DIP
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    # Ring: MCP, PIP, DIP
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    # Pinky: MCP, PIP, DIP
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
)
_SPLAY_FINGER_RAYS = (
    (2, 4),   # thumb ray
    (5, 8),   # index ray
    (9, 12),  # middle ray
    (13, 16), # ring ray
    (17, 20), # pinky ray
)


def canonicalize_saved_crop_bgr(
    crop_bgr: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return the canonical UNFLIPPED camera crop BGR array.

    - Snapshots saved via `diagnostic_storage.py` or `capture_webcam_collection.py` already store
      `crop_orientation` starting with `'canonical_unflipped'` — returned as-is.
    - Legacy snapshots saved before `diagnostic_storage.py` with `flip_input_for_model=True`
      had their saved `received_crop.png` pre-flipped by the browser canvas — flipped back once
      here so downstream `legacy` vs `method_3b` pipelines never double-flip.
    """
    if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError("crop_bgr must be a non-empty 3-channel BGR image.")

    meta = metadata or {}
    orientation = str(meta.get("crop_orientation", "") or "").strip()
    if orientation.startswith("canonical_unflipped"):
        return crop_bgr
    if bool(meta.get("flip_input_for_model", False)):
        return cv2.flip(crop_bgr, 1)
    return crop_bgr


def extract_webcam_hog_feature(
    crop_bgr: np.ndarray,
    preprocess_mode: str = "method_3b",
    metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 1764-dim HOG features identically to `SVMSignPredictor.predict_image_bgr`.

    Args:
        crop_bgr: BGR image from `received_crop.png` or live camera crop.
        preprocess_mode:
            - `'legacy'`: canonical unflipped crop -> 64x64 white-padded -> gray -> 1764 HOG.
            - `'method_3b'`: canonical unflipped crop -> single horizontal flip + soft background
              suppression preserving hand edges -> 64x64 white-padded -> gray -> 1764 HOG.
        metadata: Optional `diagnostic_metadata.json` dict for orientation canonicalization.

    Returns:
        `(hog_features_float32_1764, gray_uint8_64x64)`.
    """
    if preprocess_mode not in ("legacy", "method_3b"):
        raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode!r}. Use 'legacy' or 'method_3b'.")

    canonical_bgr = canonicalize_saved_crop_bgr(crop_bgr, metadata=metadata)
    if preprocess_mode == "method_3b":
        model_input_bgr, _, _ = apply_webcam_method_3b_from_bgr(canonical_bgr, already_flipped=False)
    else:
        model_input_bgr = canonical_bgr

    out = extract_hog_with_diagnostics_from_bgr(model_input_bgr)
    if out is None:
        raise ValueError("Failed to extract HOG features from BGR image.")
    hog_vec, _resized_bgr, gray_64 = out
    if hog_vec.shape != (HOG_FEATURE_DIM,):
        raise RuntimeError(f"Expected {HOG_FEATURE_DIM} HOG features, got {hog_vec.shape}")
    return hog_vec.astype(np.float32), gray_64


def _extract_xy_point(point) -> tuple[float, float]:
    """Read (x, y) from either a MediaPipe Landmark object or a frontend HandFrame {x, y} dict."""
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point.x), float(point.y)


def normalize_hand_landmarks_xy(
    landmarks,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> np.ndarray:
    """Convert 21 MediaPipe hand landmarks into an aspect-ratio-consistent 42-dim vector.

    Formula (identical for training crops and live frontend `HandFrame.hands`):
    1. Take normalized `(x_i, y_i)` for `i = 0..20` (supports MediaPipe Landmark objects
       and plain `{x, y, z?}` dicts from frontend `HandFrame.hands`).
    2. Convert to isotropic pixel-scale coordinates using `(image_width, image_height)`:
       `px_i = x_i * image_width`, `py_i = y_i * image_height`.
       This guarantees identical geometry between rectangular crops (e.g. 119x243) and
       live webcam frames (e.g. 640x480 or 1280x720).
    3. Translate by wrist landmark 0: `dx_i = px_i - px_0`, `dy_i = py_i - py_0`.
    4. Compute `scale = max_i sqrt(dx_i^2 + dy_i^2)` (fallback `1.0` if degenerate < 1e-6).
    5. Return `[dx_0/scale, dy_0/scale, ..., dx_20/scale, dy_20/scale]` as `float32` shape `(42,)`.
    """
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, got {len(landmarks)}")
    w = float(image_width)
    h = float(image_height)
    if w <= 0 or h <= 0 or not np.isfinite(w) or not np.isfinite(h):
        raise ValueError(f"image_width and image_height must be positive finite numbers, got ({w}, {h})")

    coords = np.array(
        [[px * w, py * h] for px, py in (_extract_xy_point(p) for p in landmarks)],
        dtype=np.float32,
    )
    rel = coords - coords[0]
    dists = np.linalg.norm(rel, axis=1)
    scale = float(np.max(dists))
    if scale < 1e-6:
        scale = 1.0
    normed = (rel / scale).reshape(-1).astype(np.float32)
    if normed.shape != (LANDMARK_FEATURE_DIM,) or not np.isfinite(normed).all():
        raise ValueError("Normalized landmark feature vector is invalid or non-finite.")
    return normed


def flip_landmark_features_horizontal(features: np.ndarray) -> np.ndarray:
    """Mirror 42-dim normalized landmark features horizontally (`dx -> -dx`, `dy` unchanged).

    Intended ONLY for training-set left/right-hand augmentation (never applied to raw cache
    or validation/test splits). Preserves all pairwise Euclidean joint distances and is an
    exact involution (`flip(flip(x)) == x`).
    """
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape != (LANDMARK_FEATURE_DIM,):
            raise ValueError(f"Expected shape ({LANDMARK_FEATURE_DIM},), got {arr.shape}")
    elif arr.ndim == 2:
        if arr.shape[1] != LANDMARK_FEATURE_DIM:
            raise ValueError(f"Expected shape (N, {LANDMARK_FEATURE_DIM}), got {arr.shape}")
    else:
        raise ValueError(f"Expected 1D or 2D landmark feature array, got ndim={arr.ndim}")

    flipped = arr.copy()
    flipped[..., 0::2] = -flipped[..., 0::2]
    # Avoid negative zero (-0.0) on wrist dx_0 for clean bitwise comparisons
    flipped[flipped == 0.0] = 0.0
    return flipped


def rotate_landmark_features_2d(
    features: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    """Rotate 42-dim wrist-normalized 2D landmarks in-plane around wrist (0, 0).

    Intended ONLY for training-set small-angle augmentation (e.g. +-12 deg), applied to
    the 21 (x, y) landmarks BEFORE extracting v2 geometric features. Preserves wrist (0, 0),
    all pairwise Euclidean distances, and all joint/splay angles.
    """
    arr = np.asarray(features, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        if arr.shape != (LANDMARK_FEATURE_DIM,):
            raise ValueError(f"Expected shape ({LANDMARK_FEATURE_DIM},), got {arr.shape}")
        pts = arr.reshape(1, 21, 2)
    elif arr.ndim == 2:
        if arr.shape[1] != LANDMARK_FEATURE_DIM:
            raise ValueError(f"Expected shape (N, {LANDMARK_FEATURE_DIM}), got {arr.shape}")
        pts = arr.reshape(-1, 21, 2)
    else:
        raise ValueError(f"Expected 1D or 2D landmark feature array, got ndim={arr.ndim}")

    rad = np.deg2rad(float(angle_degrees))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    x = pts[..., 0]
    y = pts[..., 1]
    rot = np.empty_like(pts, dtype=np.float32)
    rot[..., 0] = x * c - y * s
    rot[..., 1] = x * s + y * c
    rot[:, 0, :] = 0.0  # keep wrist strictly at (0, 0)
    rot[rot == 0.0] = 0.0
    out = rot.reshape(-1, LANDMARK_FEATURE_DIM)
    return out[0] if single else out


def _safe_cos_angle(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute bounded cosine similarity in [-1, 1] with zero-length safety (no NaN/Inf)."""
    dot = np.sum(u * v, axis=-1)
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    denom = nu * nv
    cos_val = np.where(denom > 1e-8, dot / np.maximum(denom, 1e-8), 0.0)
    return np.clip(cos_val, -1.0, 1.0).astype(np.float32)


def extract_landmark_geometric_features_v2(features_xy_42: np.ndarray) -> np.ndarray:
    """Build the 86-dim v2 geometric landmark feature vector from 42-dim normalized (x, y).

    100% derived from aspect-ratio-consistent 21 (x, y) landmarks (compatible with frontend
    `HandFrame.hands` and `asl_landmarks_cache.npz`):
      - [0:42]  42 wrist-centered, max-radius-normalized (x, y) coordinates (preserves
                absolute orientation needed to separate P/Q/G/H; no forced palm uprighting).
      - [42:52] 10 pairwise Euclidean distances between the 5 fingertips (4, 8, 12, 16, 20).
      - [52:57]  5 fingertip-to-wrist distances d(tip, 0).
      - [57:62]  5 fingertip-to-own-MCP distances d(4,2), d(8,5), d(12,9), d(16,13), d(20,17).
      - [62:67]  5 thumb-tip (4) to finger-knuckle distances d(4, 5/9/13/17/10).
      - [67:82] 15 finger joint bend cosines (3 consecutive joints per finger, 5 fingers).
      - [82:86]  4 adjacent inter-finger splay cosines between MCP->TIP finger rays.
    All 44 geometric features [42:86] are strictly invariant to translation, scale, 2D in-plane
    rotation, and horizontal mirror flip.
    """
    arr = np.asarray(features_xy_42, dtype=np.float32)
    single = arr.ndim == 1
    if single:
        if arr.shape != (LANDMARK_FEATURE_DIM,):
            raise ValueError(f"Expected shape ({LANDMARK_FEATURE_DIM},), got {arr.shape}")
        xy_2d = arr.reshape(1, LANDMARK_FEATURE_DIM)
    elif arr.ndim == 2:
        if arr.shape[1] != LANDMARK_FEATURE_DIM:
            raise ValueError(f"Expected shape (N, {LANDMARK_FEATURE_DIM}), got {arr.shape}")
        xy_2d = arr
    else:
        raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")

    pts = xy_2d.reshape(-1, 21, 2)  # (N, 21, 2)

    # Block B: 10 pairwise fingertip distances
    tip_pair_dists = [
        np.linalg.norm(pts[:, a, :] - pts[:, b, :], axis=-1)
        for a, b in _FINGERTIP_PAIRS
    ]
    # Block C1: 5 fingertip-to-wrist distances
    tip_wrist_dists = [
        np.linalg.norm(pts[:, tip, :] - pts[:, 0, :], axis=-1)
        for tip in _FINGERTIP_INDICES
    ]
    # Block C2: 5 fingertip-to-MCP distances
    tip_mcp_dists = [
        np.linalg.norm(pts[:, tip, :] - pts[:, mcp, :], axis=-1)
        for tip, mcp in _TIP_TO_MCP_PAIRS
    ]
    # Block C3: 5 thumb-tip to knuckle distances
    thumb_knuckle_dists = [
        np.linalg.norm(pts[:, a, :] - pts[:, b, :], axis=-1)
        for a, b in _THUMB_TO_KNUCKLE_PAIRS
    ]
    # Block D: 15 joint bend cosines at vertex b for triplet (a, b, c)
    joint_cosines = [
        _safe_cos_angle(pts[:, a, :] - pts[:, b, :], pts[:, c, :] - pts[:, b, :])
        for a, b, c in _JOINT_ANGLE_TRIPLETS
    ]
    # Block E: 4 adjacent inter-finger splay cosines between MCP->TIP rays
    rays = [pts[:, tip, :] - pts[:, mcp, :] for mcp, tip in _SPLAY_FINGER_RAYS]
    splay_cosines = [_safe_cos_angle(rays[i], rays[i + 1]) for i in range(len(rays) - 1)]

    geom_block = np.stack(
        tip_pair_dists + tip_wrist_dists + tip_mcp_dists + thumb_knuckle_dists + joint_cosines + splay_cosines,
        axis=1,
    ).astype(np.float32)

    out = np.concatenate([xy_2d.astype(np.float32), geom_block], axis=1)
    if out.shape[1] != LANDMARK_GEOM_V2_FEATURE_DIM or not np.isfinite(out).all():
        raise ValueError(f"Invalid v2 geometric feature matrix: shape={out.shape}")
    return out[0] if single else out


def normalize_hand_landmarks_geom_v2(
    landmarks,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> np.ndarray:
    """Convert 21 landmarks (MediaPipe or frontend HandFrame.hands dicts) directly to 86-dim v2."""
    xy_42 = normalize_hand_landmarks_xy(landmarks, image_width=image_width, image_height=image_height)
    return extract_landmark_geometric_features_v2(xy_42)


def create_hand_landmarker(
    model_path: Path | str = DEFAULT_HAND_LANDMARKER_MODEL,
    min_detection_confidence: float = 0.3,
    num_hands: int = 1,
):
    """Create a MediaPipe Tasks `HandLandmarker` in IMAGE mode for deterministic feature extraction."""
    model_p = Path(model_path)
    if not model_p.is_file():
        raise FileNotFoundError(f"HandLandmarker model not found: {model_p}")

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_p)),
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_mediapipe_landmarks_feature_from_bgr(
    image_bgr: np.ndarray,
    detector,
    metadata: dict[str, Any] | None = None,
    flip_horizontal: bool = False,
    border_pad_ratio: float = 0.5,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Extract 42-dim normalized MediaPipe hand landmarks from a BGR image (crop or full frame).

    Adding a neutral border (`border_pad_ratio=0.5`) significantly improves MediaPipe detection
    on tight hand crops (`received_crop.png`: 54/55 vs 48/55 without padding), while leaving the
    wrist-centered, scale-normalized 42-dim vector mathematically invariant to border padding.
    """
    import mediapipe as mp

    canonical_bgr = canonicalize_saved_crop_bgr(image_bgr, metadata=metadata)
    work_bgr = cv2.flip(canonical_bgr, 1) if flip_horizontal else canonical_bgr
    rgb = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2RGB)

    if border_pad_ratio > 0:
        h, w = rgb.shape[:2]
        pad = max(16, int(round(max(h, w) * border_pad_ratio)))
        rgb_input = cv2.copyMakeBorder(
            rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(232, 232, 232)
        )
    else:
        rgb_input = rgb

    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_input))
    if not result.hand_landmarks:
        return None, {"detected": False, "num_hands": 0}

    # If multiple hands exist, pick largest bounding box in actual pixel coordinates
    in_h, in_w = rgb_input.shape[:2]
    best_idx, best_hand = max(
        enumerate(result.hand_landmarks),
        key=lambda item: (
            (max(p.x for p in item[1]) - min(p.x for p in item[1])) * in_w
            * (max(p.y for p in item[1]) - min(p.y for p in item[1])) * in_h
        ),
    )
    features = normalize_hand_landmarks_xy(best_hand, image_width=in_w, image_height=in_h)
    handedness = None
    if getattr(result, "handedness", None) and best_idx < len(result.handedness) and result.handedness[best_idx]:
        handedness = str(result.handedness[best_idx][0].category_name)
    return features, {
        "detected": True,
        "num_hands": len(result.hand_landmarks),
        "mediapipe_handedness": handedness,
        "input_width": int(in_w),
        "input_height": int(in_h),
    }
