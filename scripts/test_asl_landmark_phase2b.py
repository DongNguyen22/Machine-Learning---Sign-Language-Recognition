"""Automated verification tests for Direction A — Phase 2b (86-dim Geometric Landmark SVM v2).

Verifies:
1. Geometric v2 extractor invariants: translation/scale/aspect-ratio parity with frontend
   `HandFrame.hands` dicts, exact invariance of all 44 geometric features [42:86] under
   horizontal flip and 2D in-plane rotation (+-12 deg, +-45 deg), zero NaN/Inf even on
   degenerate zero-length points, and rotation preserves wrist (0, 0) & max radius.
2. Saved v2 model bundle & metadata contract (`mediapipe_21_xy_geom_v2`, 86 dims, 36 classes,
   `Pipeline([('scaler', StandardScaler), ('clf', LinearSVC|SVC)])`, train excludes P2/P7/P8/P10,
   matching SHA-256).
3. Bit-for-bit reload parity of `predict()` and `decision_function()` on all 6,742 detected test
   samples from `asl_landmarks_cache.npz`.
4. All 7 legacy models in `models/svm/` and all v1 artifacts in `landmark_asl_v1/` remain intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.webcam_features import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    extract_landmark_geometric_features_v2,
    flip_landmark_features_horizontal,
    normalize_hand_landmarks_geom_v2,
    normalize_hand_landmarks_xy,
    rotate_landmark_features_2d,
)

V1_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v1"
V1_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1"
V2_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v2"
V2_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v2"
CACHE_PATH = V1_REPORT_DIR / "asl_landmarks_cache.npz"
V2_MODEL_PATH = V2_MODEL_DIR / "svm_landmark_asl_v2.joblib"
V2_META_PATH = V2_MODEL_DIR / "svm_landmark_asl_v2_metadata.json"

EXISTING_7_MODELS = {
    "svm_baseline_c1.joblib",
    "svm_baseline_c1_prefix_group.joblib",
    "svm_c0_01_prefix_group.joblib",
    "svm_c0_1_prefix_group.joblib",
    "svm_c10_0_prefix_group.joblib",
    "svm_c1_0_prefix_group.joblib",
    "svm_final_c0_01.joblib",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class TestASLLandmarkPhase2b(unittest.TestCase):
    def test_01_v2_invariance_and_frontend_handframe_parity(self) -> None:
        # Build a realistic non-degenerate 21-point hand in pixel coordinates
        rng = np.random.default_rng(123)
        base_px = np.zeros((21, 2), dtype=np.float64)
        for i in range(1, 21):
            base_px[i] = [np.cos(i * 0.3) * (20 + 4 * i) + rng.normal(0, 2), -abs(np.sin(i * 0.25)) * (30 + 5 * i)]

        # 1. Frontend HandFrame.hands dicts on 640x480 webcam frame (translated & scaled)
        w1, h1 = 640.0, 480.0
        px1 = base_px * 1.45 + np.array([280.0, 390.0])
        handframe_dicts = [{"x": float(p[0] / w1), "y": float(p[1] / h1), "z": 0.0} for p in px1]

        # 2. MediaPipe Landmark objects on rectangular 239x363 crop (different translation & scale)
        w2, h2 = 239.0, 363.0
        px2 = base_px * 0.72 + np.array([115.0, 290.0])
        mp_objs = [SimpleNamespace(x=float(p[0] / w2), y=float(p[1] / h2), z=0.0) for p in px2]

        v2_frontend = normalize_hand_landmarks_geom_v2(handframe_dicts, image_width=w1, image_height=h1)
        v2_crop = normalize_hand_landmarks_geom_v2(mp_objs, image_width=w2, image_height=h2)
        self.assertEqual(v2_frontend.shape, (LANDMARK_GEOM_V2_FEATURE_DIM,))
        self.assertTrue(np.isfinite(v2_frontend).all())
        np.testing.assert_allclose(v2_frontend, v2_crop, atol=1e-5)

        # 3. Horizontal flip and 2D rotations (+-12 deg, +-45 deg) MUST preserve all 44 geom features [42:86]
        xy_42 = normalize_hand_landmarks_xy(handframe_dicts, image_width=w1, image_height=h1)
        geom_ref = v2_frontend[LANDMARK_FEATURE_DIM:]

        xy_flip = flip_landmark_features_horizontal(xy_42)
        v2_flip = extract_landmark_geometric_features_v2(xy_flip)
        np.testing.assert_allclose(v2_flip[LANDMARK_FEATURE_DIM:], geom_ref, atol=1e-5)

        for deg in (-45.0, -12.0, 12.0, 45.0):
            xy_rot = rotate_landmark_features_2d(xy_42, deg)
            self.assertEqual(float(xy_rot[0]), 0.0)
            self.assertEqual(float(xy_rot[1]), 0.0)
            radii = np.linalg.norm(xy_rot.reshape(21, 2), axis=1)
            self.assertAlmostEqual(float(np.max(radii)), 1.0, places=5)
            v2_rot = extract_landmark_geometric_features_v2(xy_rot)
            self.assertTrue(np.isfinite(v2_rot).all())
            np.testing.assert_allclose(v2_rot[LANDMARK_FEATURE_DIM:], geom_ref, atol=1e-5)

        # 4. Degenerate all-zero vector produces finite 86-dim output with zero NaN/Inf
        deg_v2 = extract_landmark_geometric_features_v2(np.zeros((LANDMARK_FEATURE_DIM,), dtype=np.float32))
        self.assertEqual(deg_v2.shape, (LANDMARK_GEOM_V2_FEATURE_DIM,))
        self.assertTrue(np.isfinite(deg_v2).all())

    def test_02_v2_model_metadata_and_bitwise_reload(self) -> None:
        self.assertTrue(V2_MODEL_PATH.is_file(), f"Missing v2 model: {V2_MODEL_PATH}")
        self.assertTrue(V2_META_PATH.is_file(), f"Missing v2 metadata: {V2_META_PATH}")

        meta = json.loads(V2_META_PATH.read_text(encoding="utf-8"))
        bundle = joblib.load(V2_MODEL_PATH)
        self.assertEqual(meta["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
        self.assertEqual(meta["feature_dim"], LANDMARK_GEOM_V2_FEATURE_DIM)
        self.assertEqual(bundle["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
        self.assertEqual(int(bundle["feature_count"]), LANDMARK_GEOM_V2_FEATURE_DIM)
        self.assertEqual(len(meta["class_names"]), 36)
        self.assertEqual(meta["model_sha256"], sha256_file(V2_MODEL_PATH))

        pipe = bundle["model"]
        self.assertIsInstance(pipe, Pipeline)
        self.assertEqual(list(pipe.named_steps.keys()), ["scaler", "clf"])
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["clf"], (LinearSVC, SVC))

        train_groups = set(meta["split_config"]["train_prefix_groups"])
        self.assertEqual(train_groups, {"P1", "P3", "P4", "P5", "P6", "P9"})
        self.assertTrue(train_groups.isdisjoint({"P2", "P7", "P8", "P10"}))

        # Bit-for-bit reload parity on all 6,742 detected test samples
        test_doc = json.loads((V2_REPORT_DIR / "phase2b_test_evaluation.json").read_text(encoding="utf-8"))
        golden = test_doc["golden_reload_verification"]

        with np.load(CACHE_PATH, allow_pickle=False) as z:
            te_det = (z["new_split"] == "test") & z["detected"]
            X_te_v2 = extract_landmark_geometric_features_v2(np.asarray(z["X_landmarks"][te_det], dtype=np.float32))
            y_te = np.asarray(z["y"][te_det], dtype=np.int32)

        self.assertEqual(X_te_v2.shape, (6742, LANDMARK_GEOM_V2_FEATURE_DIM))
        y_pred = np.asarray(pipe.predict(X_te_v2), dtype=np.int32)
        scores = np.asarray(pipe.decision_function(X_te_v2), dtype=np.float64)

        n_slice = int(golden["slice_size"])
        self.assertEqual(y_pred[:n_slice].tolist(), golden["y_pred_first_64"])
        self.assertEqual(sha256_array(scores[:n_slice]), golden["decision_scores_first_64_sha256"])
        self.assertEqual(sha256_array(y_pred), golden["full_y_pred_sha256"])
        self.assertEqual(sha256_array(scores), golden["full_decision_scores_sha256"])
        self.assertEqual(int(np.sum(y_pred == y_te)), int(test_doc["correct_samples"]))

    def test_03_reports_and_v1_legacy_intact(self) -> None:
        val_doc = json.loads((V2_REPORT_DIR / "phase2b_validation_comparison.json").read_text(encoding="utf-8"))
        wb_doc = json.loads((V2_REPORT_DIR / "phase2b_webcam_3way_benchmark.json").read_text(encoding="utf-8"))
        diag_doc = json.loads((V2_REPORT_DIR / "phase2b_cd_geometry_diagnosis.json").read_text(encoding="utf-8"))

        self.assertEqual(len(val_doc["candidates_ranked"]), 18)
        self.assertEqual(wb_doc["sets"]["webcam_pilot_15"]["total_samples"], 15)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_confirmed_32"]["total_samples"], 32)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_old_23"]["total_samples"], 23)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_legacy_55"]["total_samples"], 55)
        self.assertIn("C", diag_doc["asl_processed_train_clusters"])
        self.assertIn("D", diag_doc["asl_processed_train_clusters"])

        # Legacy 7 models + v1 model intact
        top_level_models = {p.name for p in (PROJECT_ROOT / "models" / "svm").glob("*.joblib")}
        self.assertEqual(top_level_models, EXISTING_7_MODELS)
        self.assertTrue((V1_MODEL_DIR / "svm_landmark_asl_v1.joblib").is_file())
        self.assertTrue((V1_REPORT_DIR / "asl_landmarks_cache.npz").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
