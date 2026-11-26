"""Automated verification tests for Direction A — Phase 3 (A–Z 26-class vs 0–9 10-class split).

Verifies:
1. Zero leakage (prefix-group & SHA-256) and complete class coverage (26/26 A–Z and 10/10 0–9)
   across train/validation/test.
2. Both saved models (`svm_landmark_az_v3.joblib` & `svm_landmark_09_v3.joblib`) and their
   metadata match `mediapipe_21_xy_geom_v2` (86 dims), `Pipeline([('scaler', StandardScaler), ('clf', SVC)])`,
   train excludes P2/P7/P8/P10, and SHA-256 parity.
3. Bit-for-bit reload parity (`predict()` and `decision_function()`) on unseen-group `test (P2, P7)`
   and bilateral horizontal-flip symmetry on both `SVM_AZ` and `SVM_09`.
4. All 7 legacy models in `models/svm/`, `landmark_asl_v1`, and `landmark_asl_v2` remain intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_asl_landmark_svm_phase2 import load_and_verify_cache
from scripts.train_asl_landmark_svm_phase3_split import (
    DIGITS_09,
    LETTERS_AZ,
    verify_subset_zero_leakage,
)
from src.preprocess.webcam_features import (
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    extract_landmark_geometric_features_v2,
    flip_landmark_features_horizontal,
)

V1_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v1"
V2_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v2"
V3_MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v3_split"
V3_REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v3_split"
CACHE_PATH = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1" / "asl_landmarks_cache.npz"

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


class TestASLLandmarkPhase3Split(unittest.TestCase):
    def test_01_zero_leakage_subset_splits(self) -> None:
        cache = load_and_verify_cache(CACHE_PATH)
        az = verify_subset_zero_leakage(cache, "AZ_26", LETTERS_AZ)
        d09 = verify_subset_zero_leakage(cache, "09_10", DIGITS_09)
        self.assertEqual(len(az["class_names"]), 26)
        self.assertEqual(len(d09["class_names"]), 10)
        self.assertEqual(az["splits"]["test"]["total"], 5200)
        self.assertEqual(d09["splits"]["test"]["total"], 2000)

    def test_02_models_metadata_bitwise_reload_and_bilateral_symmetry(self) -> None:
        cache = load_and_verify_cache(CACHE_PATH)
        X_all = cache["X_all"]
        test_doc = json.loads((V3_REPORT_DIR / "phase3_test_evaluation.json").read_text(encoding="utf-8"))

        for fname, sub_key, classes in (
            ("svm_landmark_az_v3", "AZ_26", LETTERS_AZ),
            ("svm_landmark_09_v3", "09_10", DIGITS_09),
        ):
            m_path = V3_MODEL_DIR / f"{fname}.joblib"
            meta_path = V3_MODEL_DIR / f"{fname}_metadata.json"
            self.assertTrue(m_path.is_file())
            self.assertTrue(meta_path.is_file())

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            bundle = joblib.load(m_path)
            self.assertEqual(meta["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
            self.assertEqual(meta["feature_dim"], LANDMARK_GEOM_V2_FEATURE_DIM)
            self.assertEqual(bundle["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
            self.assertEqual(int(bundle["feature_count"]), LANDMARK_GEOM_V2_FEATURE_DIM)
            self.assertEqual(list(meta["class_names"]), classes)
            self.assertEqual(meta["model_sha256"], sha256_file(m_path))

            pipe = bundle["model"]
            self.assertIsInstance(pipe, Pipeline)
            self.assertEqual(list(pipe.named_steps.keys()), ["scaler", "clf"])
            self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
            self.assertIsInstance(pipe.named_steps["clf"], SVC)

            tr_groups = set(meta["split_config"]["train_prefix_groups"])
            self.assertEqual(tr_groups, {"P1", "P3", "P4", "P5", "P6", "P9"})
            self.assertTrue(tr_groups.isdisjoint({"P2", "P7", "P8", "P10"}))

            sub = verify_subset_zero_leakage(cache, sub_key, classes)
            X_te_xy = X_all[sub["splits"]["test"]["mask_det"]]
            X_te_v2 = extract_landmark_geometric_features_v2(X_te_xy)
            X_te_v2_mir = extract_landmark_geometric_features_v2(flip_landmark_features_horizontal(X_te_xy))

            y_pred = np.asarray(pipe.predict(X_te_v2), dtype=np.int32)
            scores = np.asarray(pipe.decision_function(X_te_v2), dtype=np.float64)
            golden = test_doc[sub_key]["golden_reload_verification"]

            n_slice = int(golden["slice_size"])
            self.assertEqual(y_pred[:n_slice].tolist(), golden["y_pred_first_64"])
            self.assertEqual(sha256_array(scores[:n_slice]), golden["decision_scores_first_64_sha256"])
            self.assertEqual(sha256_array(y_pred), golden["full_y_pred_sha256"])
            self.assertEqual(sha256_array(scores), golden["full_decision_scores_sha256"])

            # Bilateral horizontal-flip symmetry check (orig vs mirrored acc within 0.005)
            y_true_local = sub["splits"]["test"]["y_det_local"]
            y_pred_mir = np.asarray(pipe.predict(X_te_v2_mir), dtype=np.int32)
            acc_orig = float(np.mean(y_pred == y_true_local))
            acc_mir = float(np.mean(y_pred_mir == y_true_local))
            self.assertAlmostEqual(acc_orig, acc_mir, delta=0.005)

    def test_03_reports_and_prior_models_intact(self) -> None:
        wb_doc = json.loads((V3_REPORT_DIR / "phase3_webcam_4way_benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(wb_doc["sets"]["webcam_pilot_15"]["total_samples"], 15)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_confirmed_32"]["total_samples"], 32)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_old_23"]["total_samples"], 23)
        self.assertEqual(wb_doc["sets"]["webcam_diagnostics_legacy_55"]["total_samples"], 55)

        top_models = {p.name for p in (PROJECT_ROOT / "models" / "svm").glob("*.joblib")}
        self.assertEqual(top_models, EXISTING_7_MODELS)
        self.assertTrue((V1_MODEL_DIR / "svm_landmark_asl_v1.joblib").is_file())
        self.assertTrue((V2_MODEL_DIR / "svm_landmark_asl_v2.joblib").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
