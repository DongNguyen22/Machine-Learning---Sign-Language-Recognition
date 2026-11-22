"""Unit tests for invariant hand landmark normalization."""

from __future__ import annotations

import numpy as np

from src.preprocessing.coordinate_normalizer import normalize_landmarks


def make_hand_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        landmarks[index] = (
            0.02 * index,
            -0.01 * index + 0.005 * (index % 3),
            0.003 * ((index * 5) % 7),
        )
    return landmarks


def test_translation_invariance() -> None:
    landmarks = make_hand_landmarks()
    translated = landmarks + np.asarray([0.4, -0.3, 0.2], dtype=np.float32)

    normalized = normalize_landmarks(landmarks)
    normalized_translated = normalize_landmarks(translated)

    np.testing.assert_allclose(normalized_translated, normalized, atol=1e-6, rtol=0.0)


def test_scale_invariance() -> None:
    landmarks = make_hand_landmarks()
    scaled = landmarks * 2.75

    normalized = normalize_landmarks(landmarks)
    normalized_scaled = normalize_landmarks(scaled)

    np.testing.assert_allclose(normalized_scaled, normalized, atol=1e-6, rtol=0.0)


def test_zero_landmarks_do_not_produce_nan_or_inf() -> None:
    normalized = normalize_landmarks(np.zeros((21, 3), dtype=np.float32))

    assert normalized.shape == (21, 3)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
