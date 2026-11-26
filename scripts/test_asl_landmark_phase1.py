"""Automated tests for Direction A — Phase 1 (42-dim normalizer, flip augmentation, cache & leak-free split)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.webcam_features import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    flip_landmark_features_horizontal,
    normalize_hand_landmarks_xy,
)

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1"


def _make_physical_hand_px() -> np.ndarray:
    """Create a realistic 21-point 2D hand geometry in isotropic pixel units."""
    rng = np.random.default_rng(123)
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[0] = [0.0, 0.0]  # wrist
    for i in range(1, 21):
        angle = -np.pi * 0.8 + (i / 20.0) * np.pi * 0.7
        radius = 25.0 + (i % 4) * 22.0 + float(rng.uniform(-2.0, 2.0))
        pts[i] = [radius * np.cos(angle), -radius * np.sin(angle)]
    return pts


def _pairwise_distances(vec_42: np.ndarray) -> np.ndarray:
    pts = vec_42.reshape(21, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.linalg.norm(diff, axis=-1)


class Phase1LandmarkTests(unittest.TestCase):
    def test_normalizer_translation_scale_and_aspect_ratio_parity(self):
        phys = _make_physical_hand_px()

        # Case 1: Live webcam frame 640x480 with frontend HandFrame.hands dict format {x, y, z}
        fw1, fh1 = 640.0, 480.0
        shift1 = np.array([310.0, 280.0])
        scale1 = 1.0
        hand_webcam_dicts = [
            {"x": float((p[0] * scale1 + shift1[0]) / fw1), "y": float((p[1] * scale1 + shift1[1]) / fh1), "z": 0.0}
            for p in phys
        ]
        v_webcam = normalize_hand_landmarks_xy(hand_webcam_dicts, image_width=fw1, image_height=fh1)

        # Case 2: Tall rectangular padded crop 239x363 (different aspect ratio, shifted & scaled 1.75x)
        # using MediaPipe Landmark-style objects with .x / .y attributes
        fw2, fh2 = 239.0, 363.0
        shift2 = np.array([120.0, 240.0])
        scale2 = 1.75
        hand_crop_objs = [
            SimpleNamespace(x=float((p[0] * scale2 + shift2[0]) / fw2), y=float((p[1] * scale2 + shift2[1]) / fh2))
            for p in phys
        ]
        v_crop = normalize_hand_landmarks_xy(hand_crop_objs, image_width=fw2, image_height=fh2)

        self.assertEqual(v_webcam.shape, (LANDMARK_FEATURE_DIM,))
        self.assertEqual(v_webcam.dtype, np.float32)
        self.assertTrue(np.isfinite(v_webcam).all())
        # Wrist is at origin (0, 0) and max radius is 1.0
        self.assertAlmostEqual(float(v_webcam[0]), 0.0, places=6)
        self.assertAlmostEqual(float(v_webcam[1]), 0.0, places=6)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(v_webcam.reshape(21, 2), axis=1))), 1.0, places=5)
        # Exact geometric parity between live 640x480 webcam frame and 239x363 rectangular crop
        np.testing.assert_allclose(v_webcam, v_crop, atol=1e-5)

    def test_horizontal_flip_is_isometry_and_involution(self):
        phys = _make_physical_hand_px()
        hand_dicts = [{"x": float((p[0] + 300.0) / 640.0), "y": float((p[1] + 250.0) / 480.0)} for p in phys]
        v = normalize_hand_landmarks_xy(hand_dicts, image_width=640.0, image_height=480.0)
        vf = flip_landmark_features_horizontal(v)
        vff = flip_landmark_features_horizontal(vf)

        # dx negated, dy unchanged
        np.testing.assert_allclose(vf[0::2], -v[0::2], atol=1e-6)
        np.testing.assert_allclose(vf[1::2], v[1::2], atol=1e-6)
        # Double flip returns exact original vector
        np.testing.assert_array_equal(vff, v)
        # All 21x21 pairwise Euclidean distances preserved
        np.testing.assert_allclose(_pairwise_distances(v), _pairwise_distances(vf), atol=1e-5)

        # Batch (2D) check
        batch = np.stack([v, vf], axis=0)
        batch_ff = flip_landmark_features_horizontal(flip_landmark_features_horizontal(batch))
        np.testing.assert_array_equal(batch_ff, batch)

    def test_cached_artifacts_and_zero_leakage_split(self):
        cache_path = EXPERIMENT_DIR / "asl_landmarks_cache.npz"
        manifest_path = EXPERIMENT_DIR / "split_manifest.csv"
        split_summary_path = EXPERIMENT_DIR / "split_summary.json"
        leakage_path = EXPERIMENT_DIR / "asl_leakage_audit.json"
        det_summary_path = EXPERIMENT_DIR / "detection_summary_by_class.json"

        for p in (cache_path, manifest_path, split_summary_path, leakage_path, det_summary_path):
            self.assertTrue(p.is_file(), f"Missing Phase 1 artifact: {p}")

        data = np.load(cache_path, allow_pickle=False)
        X = data["X_landmarks"]
        y = data["y"]
        detected = data["detected"]
        class_names = data["class_names"]
        cached_shas = data["file_sha256"]
        self.assertEqual(str(data["feature_schema"]), LANDMARK_FEATURE_SCHEMA)
        self.assertEqual(int(data["feature_dim"]), LANDMARK_FEATURE_DIM)
        self.assertEqual(len(class_names), 36)
        self.assertEqual(X.shape, (36000, 42))
        self.assertEqual(y.shape, (36000,))
        self.assertEqual(detected.shape, (36000,))

        # No NaN/Inf on detected samples; wrist at (0,0), max L2 radius <= 1.0001
        X_det = X[detected]
        self.assertGreater(len(X_det), 0)
        self.assertTrue(np.isfinite(X_det).all())
        np.testing.assert_allclose(X_det[:, 0:2], 0.0, atol=1e-6)
        radii = np.max(np.linalg.norm(X_det.reshape(-1, 21, 2), axis=2), axis=1)
        self.assertTrue(np.all(radii <= 1.0001))

        # Verify manifest CSV matches cache row count and has 0 prefix_group & 0 SHA-256 leakage
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 36000)

        pg_splits: dict[str, set[str]] = {}
        sha_splits: dict[str, set[str]] = {}
        classes_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
        for i, r in enumerate(rows):
            sp = r["new_split"]
            self.assertIn(sp, ("train", "validation", "test"))
            pg_splits.setdefault(r["prefix_group"], set()).add(sp)
            sha_splits.setdefault(r["file_sha256"], set()).add(sp)
            classes_by_split[sp].add(r["class_name"])
            self.assertEqual(r["file_sha256"], str(cached_shas[i]))
            self.assertEqual(r["detected"].lower() == "true", bool(detected[i]))

        self.assertEqual([pg for pg, s in pg_splits.items() if len(s) > 1], [])
        self.assertEqual([h for h, s in sha_splits.items() if len(s) > 1], [])
        for sp in ("train", "validation", "test"):
            self.assertEqual(len(classes_by_split[sp]), 36)

        split_sum = json.loads(split_summary_path.read_text(encoding="utf-8"))
        self.assertTrue(split_sum["leakage_verification"]["zero_prefix_group_leakage"])
        self.assertTrue(split_sum["leakage_verification"]["zero_sha256_leakage"])


if __name__ == "__main__":
    unittest.main()
