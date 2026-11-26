"""Automated unit tests for batch multi-label capture helper and 36-class webcam benchmark script.

Runs strictly inside `tempfile.TemporaryDirectory()` — NEVER touches `data/webcam_eval_36`,
any existing webcam dataset, or any saved model.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from capture_webcam_collection import parse_labels_spec, save_sample
from evaluate_webcam_36_benchmark import main as eval_36_main
from manage_webcam_samples import load_saved_samples, update_sample_label
from src.model.predict_landmark_v2 import (
    DEFAULT_LANDMARK_09_MODEL_PATH,
    DEFAULT_LANDMARK_AZ_MODEL_PATH,
    DEFAULT_LANDMARK_V2_MODEL_PATH,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.preprocess.webcam_features import (
    canonicalize_saved_crop_bgr,
    create_hand_landmarker,
    extract_landmark_geometric_features_v2,
    extract_mediapipe_landmarks_feature_from_bgr,
)

PILOT_DIR = PROJECT_ROOT / "data" / "webcam_pilot"
SIGN_5_REF = PROJECT_ROOT / "frontend" / "public" / "signs" / "5.jpg"


class TestWebcamEval36(unittest.TestCase):
    def test_01_parse_labels_spec_ranges_lists_and_validation(self) -> None:
        self.assertEqual(parse_labels_spec("a", None), ["A"])
        self.assertEqual(parse_labels_spec(None, "0-9"), [str(i) for i in range(10)])
        self.assertEqual(parse_labels_spec(None, "A-M"), [chr(c) for c in range(ord("A"), ord("M") + 1)])
        self.assertEqual(parse_labels_spec(None, "N-Z"), [chr(c) for c in range(ord("N"), ord("Z") + 1)])
        self.assertEqual(len(parse_labels_spec(None, "ALL")), 36)
        self.assertEqual(parse_labels_spec(None, "0,1,A,A,B"), ["0", "1", "A", "B"])

        with self.assertRaises(ValueError):
            parse_labels_spec("A", "A-C")
        with self.assertRaises(ValueError):
            parse_labels_spec(None, None)
        with self.assertRaises(ValueError):
            parse_labels_spec(None, "9-0")
        with self.assertRaises(ValueError):
            parse_labels_spec(None, "0-A")
        with self.assertRaises(ValueError):
            parse_labels_spec(None, "ZZ")

    def test_02_eval_refuses_unconfirmed_labels(self) -> None:
        pilot_folders = sorted(p for p in PILOT_DIR.iterdir() if p.is_dir())
        crop_bgr = cv2.imread(str(pilot_folders[0] / "received_crop.png"))
        self.assertIsNotNone(crop_bgr)

        with tempfile.TemporaryDirectory(prefix="eval36_refuse_") as tmp:
            root = Path(tmp)
            samples_dir = root / "samples"
            out_dir = root / "out"

            # Save 2 samples; confirm only 1 -> MUST refuse official evaluation!
            save_sample(
                samples_dir,
                crop_bgr,
                {
                    "participant_id": "P01",
                    "session_id": "S01",
                    "gesture_instance_id": "A_G01",
                    "handedness": "left",
                    "background": "plain",
                    "lighting": "daylight",
                    "intended_sign": "A",
                },
            )
            save_sample(
                samples_dir,
                crop_bgr,
                {
                    "participant_id": "P01",
                    "session_id": "S01",
                    "gesture_instance_id": "A_G02",
                    "handedness": "left",
                    "background": "plain",
                    "lighting": "daylight",
                    "intended_sign": "A",
                },
            )
            loaded = load_saved_samples(samples_dir)
            update_sample_label(loaded[0], "A", confirmed=True)

            rc = eval_36_main(["--samples-dir", str(samples_dir), "--output-dir", str(out_dir)])
            self.assertEqual(rc, 2)
            self.assertFalse((out_dir / "webcam_eval_36_report.json").exists())

    def test_03_eval_e2e_vs_conditional_and_exact_4model_parity(self) -> None:
        # Pick 1 real A and 1 real B from webcam_pilot, 1 real digit 5, and 1 blank image (no hand)
        pilot_a = None
        pilot_b = None
        for p in sorted(d for d in PILOT_DIR.iterdir() if d.is_dir()):
            meta = json.loads((p / "diagnostic_metadata.json").read_text(encoding="utf-8"))
            lbl = str(meta.get("verified_true_label", ""))
            if lbl == "A" and pilot_a is None:
                pilot_a = cv2.imread(str(p / "received_crop.png"))
            elif lbl == "B" and pilot_b is None:
                pilot_b = cv2.imread(str(p / "received_crop.png"))
        digit_5 = cv2.imread(str(SIGN_5_REF))
        blank_no_hand = np.full((140, 140, 3), 220, dtype=np.uint8)
        self.assertIsNotNone(pilot_a)
        self.assertIsNotNone(pilot_b)
        self.assertIsNotNone(digit_5)

        with tempfile.TemporaryDirectory(prefix="eval36_ok_") as tmp:
            root = Path(tmp)
            samples_dir = root / "samples"
            out_dir = root / "out"

            fixtures = [
                ("A", "A_G01", pilot_a),
                ("B", "B_G01", pilot_b),
                ("5", "5_G01", digit_5),
                ("C", "C_G01", blank_no_hand),  # guaranteed hand_detected=False
            ]
            for lbl, gid, img in fixtures:
                save_sample(
                    samples_dir,
                    img,
                    {
                        "participant_id": "P_TEST",
                        "session_id": "S01",
                        "gesture_instance_id": gid,
                        "handedness": "left",
                        "background": "plain",
                        "lighting": "daylight",
                        "intended_sign": lbl,
                    },
                )
            for s in load_saved_samples(samples_dir):
                update_sample_label(s, str(s["current_label"]), confirmed=True)

            rc = eval_36_main(["--samples-dir", str(samples_dir), "--output-dir", str(out_dir)])
            self.assertEqual(rc, 0)

            report_path = out_dir / "webcam_eval_36_report.json"
            self.assertTrue(report_path.is_file())
            self.assertTrue((out_dir / "webcam_eval_36_per_class.csv").is_file())
            self.assertTrue((out_dir / "webcam_eval_36_samples.csv").is_file())
            self.assertTrue((out_dir / "webcam_eval_36_errors.csv").is_file())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            aud = report["dataset_audit"]
            self.assertEqual(aud["total_samples"], 4)
            self.assertEqual(aud["classes_covered_count"], 4)
            self.assertEqual(aud["distinct_participants_count"], 1)
            # Honest limitation notes MUST flag < 3 participants and < 36 classes
            self.assertEqual(len(aud["limitation_notes"]), 2)

            # Check End-to-End (denominator 4) vs Conditional (denominator 3 detected)
            all36 = report["summary"]["all_36"]["landmark_all36"]
            self.assertEqual(all36["total_samples"], 4)
            self.assertEqual(all36["detected_samples"], 3)
            self.assertEqual(all36["undetected_samples"], 1)
            self.assertAlmostEqual(
                all36["end_to_end_accuracy_pct"],
                round(all36["correct_e2e"] / 4 * 100.0, 2),
                places=2,
            )
            self.assertAlmostEqual(
                all36["conditional_accuracy_pct"],
                round(all36["correct_on_detected"] / 3 * 100.0, 2),
                places=2,
            )

            # Verify bit-for-bit prediction parity with direct 4 models
            hog_pred = SVMSignPredictor(PROJECT_ROOT / DEFAULT_MODEL_PATH)
            b36 = joblib.load(DEFAULT_LANDMARK_V2_MODEL_PATH)
            baz = joblib.load(DEFAULT_LANDMARK_AZ_MODEL_PATH)
            b09 = joblib.load(DEFAULT_LANDMARK_09_MODEL_PATH)
            det = create_hand_landmarker(min_detection_confidence=0.3, num_hands=1)
            try:
                import csv

                with (out_dir / "webcam_eval_36_samples.csv").open(encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 4)
                for row in rows:
                    sdir = samples_dir / row["folder_name"]
                    bgr = cv2.imread(str(sdir / "received_crop.png"))
                    meta = json.loads((sdir / "diagnostic_metadata.json").read_text(encoding="utf-8"))
                    can = canonicalize_saved_crop_bgr(bgr, metadata=meta)
                    direct_hog = hog_pred.predict_image_bgr(can, top_k=3, experimental_webcam_mode=True)
                    self.assertEqual(row["hog_3b_pred"], direct_hog["predicted_label"])

                    f42, _ = extract_mediapipe_landmarks_feature_from_bgr(
                        bgr, detector=det, metadata=meta, flip_horizontal=False, border_pad_ratio=0.5
                    )
                    if f42 is None:
                        self.assertEqual(row["hand_detected"], "False")
                        self.assertEqual(row["landmark_all36_pred"], "__NO_HAND__")
                        self.assertEqual(row["landmark_scoped_pred"], "__NO_HAND__")
                    else:
                        self.assertEqual(row["hand_detected"], "True")
                        f86 = extract_landmark_geometric_features_v2(f42).reshape(1, -1)
                        p36 = str(b36["class_names"][int(b36["model"].predict(f86)[0])])
                        self.assertEqual(row["landmark_all36_pred"], p36)
                        if row["true_label"].isdigit():
                            p09 = str(b09["class_names"][int(b09["model"].predict(f86)[0])])
                            self.assertEqual(row["landmark_scoped_pred"], p09)
                        else:
                            paz = str(baz["class_names"][int(baz["model"].predict(f86)[0])])
                            self.assertEqual(row["landmark_scoped_pred"], paz)
            finally:
                det.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
