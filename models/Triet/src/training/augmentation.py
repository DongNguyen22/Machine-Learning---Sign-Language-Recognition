"""Train-only augmentation for 63/91-dimensional ASL feature vectors."""

from __future__ import annotations

import numpy as np

from src.feature_extraction.geometric_features import extract_geometric_features
from src.preprocessing.coordinate_normalizer import normalize_landmarks


def augment_landmark_samples(
    X: np.ndarray,
    y: np.ndarray,
    feature_mode: str = "91d",
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Tăng cường train rồi trích lại 91 đặc trưng để augumentation không làm sai hình học bàn tay."""
    features = np.asarray(X, dtype=np.float32)
    labels = np.asarray(y).copy()
    mode = str(feature_mode).lower()

    if features.ndim != 2:
        raise ValueError("X phải có shape (N, 63) hoặc (N, 91).")
    if len(features) != len(labels):
        raise ValueError("X và y phải có cùng số mẫu.")
    if mode not in ("63d", "91d"):
        raise ValueError("feature_mode phải là '63d' hoặc '91d'.")
    expected_width = 63 if mode == "63d" else 91
    if features.shape[1] != expected_width:
        raise ValueError(
            f"feature_mode={feature_mode} yêu cầu X có {expected_width} cột."
        )
    if not np.isfinite(features).all():
        raise ValueError("X không được chứa NaN hoặc Inf.")

    rng = np.random.default_rng(random_seed)
    landmarks = features[:, :63].reshape(-1, 21, 3).copy()

    for index, sample in enumerate(landmarks):
        sample[0] = np.zeros(3, dtype=np.float32)

        degrees = float(rng.uniform(-15.0, 15.0))
        theta = np.deg2rad(degrees)
        cosine = np.cos(theta)
        sine = np.sin(theta)
        x_values = sample[:, 0].copy()
        y_values = sample[:, 1].copy()
        sample[:, 0] = cosine * x_values - sine * y_values
        sample[:, 1] = sine * x_values + cosine * y_values

        if rng.random() < 0.5:
            sample[:, 0] *= -1.0

        sample += rng.normal(0.0, 0.01, size=(21, 3)).astype(np.float32)
        sample[0] = np.zeros(3, dtype=np.float32)

    augmented: list[np.ndarray] = []
    for sample in landmarks:
        normalized = normalize_landmarks(sample)
        if mode == "63d":
            result = normalized.reshape(63).astype(np.float32)
        else:
            result = extract_geometric_features(normalized, baseline_63=False)
        if not np.isfinite(result).all():
            raise ValueError("Augmentation tạo ra giá trị NaN hoặc Inf.")
        augmented.append(result)

    if not augmented:
        empty_width = 63 if mode == "63d" else 91
        return np.empty((0, empty_width), dtype=np.float32), labels.copy()
    return np.asarray(augmented, dtype=np.float32), labels.copy()


def augment_train_coordinates(
    features: np.ndarray,
    seed: int | None = None,
    rotate: bool = True,
    noise: bool = True,
    horizontal_flip: bool = True,
    flip_probability: float = 0.5,
) -> np.ndarray:
    """Augment coordinate blocks only; never call this function for val/test."""
    x = np.asarray(features, dtype=np.float32)
    mode = "63d" if x.shape[1] == 63 else "91d"
    dummy_y = np.zeros(len(x), dtype=np.int32)
    augmented, _ = augment_landmark_samples(
        x,
        dummy_y,
        feature_mode=mode,
        random_seed=seed if seed is not None else 42,
    )
    return augmented


augment_coordinates = augment_train_coordinates
