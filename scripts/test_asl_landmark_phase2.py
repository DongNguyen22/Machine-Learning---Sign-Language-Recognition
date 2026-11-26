"""Automated verification tests for Direction A — Phase 2 (Landmark SVM on asl_processed).

Verifies:
1. Saved joblib model reloads cleanly and produces bit-for-bit identical `predict()` and
   `decision_function()` outputs against the golden test outputs recorded right after training.
2. Model bundle & metadata JSON strictly match `LANDMARK_FEATURE_SCHEMA`, 42 dims, 36 classes,
   `Pipeline([('scaler', StandardScaler), ('clf', LinearSVC|SVC)])` (never RandomForest),
   zero leakage in train prefix groups (excludes P2, P7, P8, P10), and matching SHA-256.
3. Phase 2 validation, test, and webcam benchmark reports are complete and internally consistent,
   and all 7 pre-existing models in `models/svm/` remain intact.
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
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.webcam_features import (
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
)

MODEL_DIR = PROJECT_ROOT / "models" / "svm" / "experiments" / "landmark_asl_v1"
MODEL_PATH = MODEL_DIR / "svm_landmark_asl_v1.joblib"
METADATA_PATH = MODEL_DIR / "svm_landmark_asl_v1_metadata.json"
REPORT_DIR = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1"
CACHE_PATH = REPORT_DIR / "asl_landmarks_cache.npz"

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


class TestASLLandmarkPhase2(unittest.TestCase):
    def test_01_metadata_and_pipeline_contract(self) -> None:
        self.assertTrue(MODEL_PATH.is_file(), f"Missing model: {MODEL_PATH}")
        self.assertTrue(METADATA_PATH.is_file(), f"Missing metadata: {METADATA_PATH}")

        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        bundle = joblib.load(MODEL_PATH)

        # Schema & dimensions
        self.assertEqual(meta["feature_schema"], LANDMARK_FEATURE_SCHEMA)
        self.assertEqual(meta["feature_dim"], LANDMARK_FEATURE_DIM)
        self.assertEqual(bundle["feature_schema"], LANDMARK_FEATURE_SCHEMA)
        self.assertEqual(int(bundle["feature_count"]), LANDMARK_FEATURE_DIM)
        self.assertAlmostEqual(float(meta["border_pad_ratio"]), 0.5, places=6)
        self.assertAlmostEqual(float(meta["min_detection_confidence"]), 0.3, places=6)

        # 36 classes
        self.assertEqual(len(meta["class_names"]), 36)
        self.assertEqual(len(bundle["class_names"]), 36)
        self.assertEqual(list(meta["class_names"]), [str(c) for c in bundle["class_names"]])

        # Pipeline must be StandardScaler + LinearSVC or SVC (never RandomForest)
        pipe = bundle["model"]
        self.assertIsInstance(pipe, Pipeline)
        self.assertEqual(list(pipe.named_steps.keys()), ["scaler", "clf"])
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["clf"], (LinearSVC, SVC))

        # No leakage in train prefix groups
        train_groups = set(meta["split_config"]["train_prefix_groups"])
        self.assertEqual(train_groups, {"P1", "P3", "P4", "P5", "P6", "P9"})
        self.assertTrue(train_groups.isdisjoint({"P2", "P7", "P8", "P10"}))

        # Model SHA-256 parity
        self.assertEqual(meta["model_sha256"], sha256_file(MODEL_PATH))

    def test_02_joblib_reload_bitwise_parity_on_test_cache(self) -> None:
        test_report = json.loads((REPORT_DIR / "phase2_test_evaluation.json").read_text(encoding="utf-8"))
        golden = test_report["golden_reload_verification"]

        bundle = joblib.load(MODEL_PATH)
        pipe: Pipeline = bundle["model"]

        with np.load(CACHE_PATH, allow_pickle=False) as z:
            te_det = (z["new_split"] == "test") & z["detected"]
            X_te = np.asarray(z["X_landmarks"][te_det], dtype=np.float32)
            y_te = np.asarray(z["y"][te_det], dtype=np.int32)

        self.assertEqual(X_te.shape, (6742, 42))
        y_pred = np.asarray(pipe.predict(X_te), dtype=np.int32)
        scores = np.asarray(pipe.decision_function(X_te), dtype=np.float64)

        # Bit-for-bit parity on both first 64 slice and entire 6,742 detected test set
        n_slice = int(golden["slice_size"])
        self.assertEqual(y_pred[:n_slice].tolist(), golden["y_pred_first_64"])
        self.assertEqual(sha256_array(scores[:n_slice]), golden["decision_scores_first_64_sha256"])
        self.assertEqual(sha256_array(y_pred), golden["full_y_pred_sha256"])
        self.assertEqual(sha256_array(scores), golden["full_decision_scores_sha256"])

        # Exact accuracy agreement with saved test evaluation JSON
        self.assertEqual(int(np.sum(y_pred == y_te)), int(test_report["correct_samples"]))
        self.assertAlmostEqual(
            float(np.mean(y_pred == y_te)),
            float(test_report["cond_accuracy"]),
            places=6,
        )

    def test_03_reports_consistency_and_existing_models_intact(self) -> None:
        val_report = json.loads((REPORT_DIR / "phase2_validation_comparison.json").read_text(encoding="utf-8"))
        test_report = json.loads((REPORT_DIR / "phase2_test_evaluation.json").read_text(encoding="utf-8"))
        webcam_report = json.loads((REPORT_DIR / "phase2_webcam_benchmark.json").read_text(encoding="utf-8"))

        self.assertEqual(len(val_report["candidates_ranked"]), 18)
        self.assertEqual(val_report["validation_prefix_groups"], ["P8", "P10"])
        self.assertEqual(test_report["prefix_groups"], ["P2", "P7"])
        self.assertEqual(test_report["total_samples"], 7200)
        self.assertEqual(test_report["detected_samples"], 6742)
        self.assertEqual(len(test_report["per_class"]), 36)
        self.assertEqual(len(test_report["top_10_confusions"]), 10)

        # Strict webcam set sizes (no mixing, no auto-confirming)
        sets = webcam_report["sets"]
        self.assertEqual(sets["webcam_pilot_15"]["total_samples"], 15)
        self.assertEqual(sets["webcam_diagnostics_confirmed_32"]["total_samples"], 32)
        self.assertEqual(sets["webcam_diagnostics_old_23"]["total_samples"], 23)
        self.assertEqual(sets["webcam_diagnostics_legacy_55"]["total_samples"], 55)

        # All 7 pre-existing models in models/svm/ intact
        top_level_models = {p.name for p in (PROJECT_ROOT / "models" / "svm").glob("*.joblib")}
        self.assertEqual(top_level_models, EXISTING_7_MODELS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
