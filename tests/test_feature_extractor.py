"""Unit tests for geometric ASL feature extraction."""

from __future__ import annotations

import numpy as np

from src.feature_extraction.geometric_features import (
    calculate_joint_angle,
    extract_geometric_features,
)


def make_hand_landmarks() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    for index in range(21):
        landmarks[index] = (
            0.02 * (index % 5),
            0.03 * (index // 5),
            0.01 * ((index * 3) % 4),
        )
    return landmarks


def test_feature_output_shapes_and_dtype() -> None:
    landmarks = make_hand_landmarks()

    baseline = extract_geometric_features(landmarks, baseline_63=True)
    full = extract_geometric_features(landmarks)

    assert baseline.shape == (63,)
    assert baseline.dtype == np.float32
    assert full.shape == (91,)
    assert full.dtype == np.float32
    assert np.isfinite(baseline).all()
    assert np.isfinite(full).all()


def test_collinear_three_points_angle_is_pi() -> None:
    left = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    vertex = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    angle = calculate_joint_angle(left, vertex, right)

    assert np.isclose(angle, np.pi, atol=1e-6)


def test_perpendicular_three_points_angle_is_pi_over_two() -> None:
    left = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    vertex = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    angle = calculate_joint_angle(left, vertex, right)

    assert np.isclose(angle, np.pi / 2.0, atol=1e-6)
