"""End-to-end automated verification on TEMPORARY synthetic fixtures (no real data/models touched).

Explicit note:
- Uses synthetic images strictly in `tempfile.TemporaryDirectory()` to verify software plumbing
  (HTTP API, atomic save, review/audit, near-duplicate warnings, leak-free split, HOG LinearSVC
  pipeline parity, and separate Landmark SVM schema isolation).
- NEVER saves synthetic models into `models/` and NEVER treats synthetic accuracy as proof of
  real webcam recognition quality.
"""

from __future__ import annotations

import copy
import csv
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from capture_webcam_collection import save_sample
from manage_webcam_samples import (
    export_collection_audit,
    load_saved_samples,
    main as manage_main,
    summarize_collection_progress,
)
from split_webcam_collection import (
    auto_assign_participants,
    main as split_main,
    validate as validate_split,
)
from train_landmark_svm_experiment import main as train_landmark_main
from train_webcam_svm import main as train_hog_main
from verify_single_sample import verify_sample_folder
from src.model.predict_landmark_svm import LandmarkSVMPredictor
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.model.svm_api_server import create_request_handler
from src.preprocess.webcam_features import (
    HOG_FEATURE_DIM,
    LANDMARK_FEATURE_DIM,
    LANDMARK_FEATURE_SCHEMA,
    extract_webcam_hog_feature,
    normalize_hand_landmarks_xy,
)


def make_synthetic_pattern_bgr(seed: int, class_idx: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(190, 235, (120, 100, 3), dtype=np.uint8)
    # Draw distinct simple geometric shape per class so synthetic LinearSVC test is well-posed
    if class_idx == 0:
        cv2.circle(img, (50, 60), 28, (60, 80, 160), -1)
    elif class_idx == 1:
        cv2.rectangle(img, (25, 20), (75, 100), (50, 120, 150), -1)
    else:
        pts = np.array([[50, 15], [20, 100], [80, 100]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (80, 70, 140))
    return img


def run_all_checks() -> None:
    predictor = SVMSignPredictor()

    with tempfile.TemporaryDirectory(prefix="signvision_e2e_") as tmp:
        root = Path(tmp)
        samples_dir = root / "samples"

        # 1. Verify real HTTP server (/health, /predict, /diagnostic/save) in temp dir
        app = SimpleNamespace(
            allowed_origins=set(),
            max_payload_bytes=4_000_000,
            max_image_bytes=3_000_000,
            predictor=predictor,
            diagnostic_save_dir=root / "http_samples",
            startup_time=0.0,
            model_path=DEFAULT_MODEL_PATH,
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_request_handler(app))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/health") as r:
                health = json.load(r)
            assert health["model_ready"] is True and health["feature_count"] == 1764 and health["class_count"] == 36
            assert health["diagnostic_save_dir"] == str(root / "http_samples")

            # Verify /diagnostic/next-instance starts at A_G01 for empty folder
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.server_port}/diagnostic/next-instance?participant_id=P_HTTP&session_id=S01&label=A"
            ) as r:
                next_before = json.load(r)
            assert next_before["next_gesture_instance_id"] == "A_G01"
            assert next_before["existing_instance_count"] == 0

            crop = make_synthetic_pattern_bgr(100, 0)
            img_url = predictor.encode_png_data_url(crop)
            req_pred = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/predict",
                data=json.dumps(
                    {
                        "image": img_url,
                        "include_diagnostics": True,
                        "frame_id": "E2E_HTTP_1",
                        "experimental_webcam_mode": True,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req_pred) as r:
                pred_res = json.load(r)
            assert pred_res["frame_id"] == "E2E_HTTP_1"

            # Centered crop -> NO border touch warning
            save_payload = {
                "participant_id": "P_HTTP",
                "session_id": "S01",
                "gesture_instance_id": "A_G01",
                "handedness": "right",
                "background": "synthetic_test",
                "lighting": "synthetic_test",
                "intended_sign": "A",
                "frame_id": "E2E_HTTP_1",
                "crop_box": {"x": 200, "y": 120, "width": 190, "height": 190},
                "source_frame_dimensions": {"width": 640, "height": 480},
                "flip_input_for_model": False,
                "preview_mirrored": True,
                "experimental_webcam_mode": True,
                "webcam_mode": "method_3b",
                "received_crop_png_data_url": pred_res["diagnostics"]["received_crop_png_data_url"],
                "preprocessed_gray_64x64_png_data_url": pred_res["diagnostics"]["preprocessed_gray_64x64_png_data_url"],
                "same_frame_comparison": {
                    k: {**v, "frame_id": "E2E_HTTP_1"}
                    for k, v in pred_res["same_frame_comparison"].items()
                },
            }
            req_save = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/diagnostic/save",
                data=json.dumps(save_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req_save) as r:
                save_res = json.load(r)
            saved_http_dir = Path(save_res["sample_directory"])
            assert save_res.get("near_duplicate_warnings") == []
            assert save_res.get("touches_frame_border") == []
            assert save_res.get("border_touch_warning") is False
            assert save_res.get("next_suggested_gesture_instance_id") == "A_G02"
            ver = verify_sample_folder(saved_http_dir, check_landmarks=False)
            assert ver["passed"] is True, ver

            # Save a second sample with right-edge touching crop_box (like pilot #11 C_G03: 332+308=640)
            save_payload_dup = copy.deepcopy(save_payload)
            save_payload_dup["gesture_instance_id"] = "A_G02"
            save_payload_dup["crop_box"] = {"x": 332, "y": 153, "width": 308, "height": 310}
            req_save2 = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/diagnostic/save",
                data=json.dumps(save_payload_dup).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req_save2) as r:
                save_res2 = json.load(r)
            dups = save_res2.get("near_duplicate_warnings", [])
            assert len(dups) >= 1 and dups[0]["exact_sha256_match"] is True, save_res2
            assert save_res2.get("touches_frame_border") == ["right"], save_res2
            assert save_res2.get("border_touch_warning") is True
            assert save_res2.get("next_suggested_gesture_instance_id") == "A_G03"
            assert saved_http_dir.is_dir() and Path(save_res2["sample_directory"]).is_dir()

            # Verify progress command reports P_HTTP accurately (0/36 completed, A in_progress 2/5, 1 border touch, 1 near-dup)
            prog = summarize_collection_progress(
                load_saved_samples(root / "http_samples"),
                participant="P_HTTP",
                target_instances=5,
            )
            assert prog["total_samples"] == 2
            assert prog["confirmed_samples"] == 0 and prog["unconfirmed_samples"] == 2
            assert prog["classes_completed_count"] == 0
            assert prog["in_progress_classes"] == ["A"]
            assert prog["per_class"]["A"]["distinct_gesture_instances"] == 2
            assert prog["border_touch_warning_count"] == 1
            assert prog["near_duplicate_warning_count"] == 1
            assert manage_main(["progress", "--diag-dir", str(root / "http_samples"), "--participant", "P_HTTP"]) == 0
            print("PASS [1/6] Real HTTP /health + /predict + /diagnostic/save (border touch + near-dup + next-instance) + progress")
        finally:
            server.shutdown()
            server.server_close()

        # 2. Verify CLI capture -> exact HOG parity with SVMSignPredictor -> near-duplicate warning
        classes = ["A", "B", "C"]
        participants = ["P01", "P02", "P03"]
        seed_counter = 1
        for p_idx, pid in enumerate(participants):
            for c_idx, lbl in enumerate(classes):
                for inst in range(1, 3):
                    img = make_synthetic_pattern_bgr(seed_counter, c_idx)
                    seed_counter += 1
                    folder = save_sample(
                        samples_dir,
                        img,
                        {
                            "participant_id": pid,
                            "session_id": "S01",
                            "gesture_instance_id": f"{lbl}_G0{inst}",
                            "handedness": "right",
                            "background": "plain_wall",
                            "lighting": "daylight",
                            "intended_sign": lbl,
                        },
                        roi_mode="manual",
                    )
                    # Bitwise parity check between training feature extractor and live SVMSignPredictor
                    hog_train, g64_train = extract_webcam_hog_feature(
                        img, preprocess_mode="method_3b", metadata={"crop_orientation": "canonical_unflipped_raw_camera"}
                    )
                    live_res = predictor.predict_image_bgr(
                        img, return_features=True, experimental_webcam_mode=True, already_flipped=False
                    )
                    assert np.array_equal(hog_train, live_res["hog_features"])
                    assert hog_train.shape == (HOG_FEATURE_DIM,)
                    disk_m3b = cv2.imread(str(folder / "method_3b_gray_64x64.png"), cv2.IMREAD_GRAYSCALE)
                    assert np.array_equal(g64_train, disk_m3b)
        print("PASS [2/6] CLI capture matches SVMSignPredictor 1764-HOG & gray64 100% bit-for-bit")

        # 3. Verify review/confirm workflow and audit without deleting or auto-relabeling
        samples_before = load_saved_samples(samples_dir)
        assert len(samples_before) == 18 and all(not s["label_confirmed"] for s in samples_before)
        manage_main(["confirm-all", "--diag-dir", str(samples_dir)])
        samples_after = load_saved_samples(samples_dir)
        assert len(samples_after) == 18 and all(s["label_confirmed"] for s in samples_after)
        audit_dir = root / "audit"
        export_collection_audit(samples_after, audit_dir)
        audit_json = json.loads((audit_dir / "collection_audit.json").read_text(encoding="utf-8"))
        assert audit_json["confirmed_samples"] == 18
        assert len(audit_json["missing_confirmed_classes"]) == 33
        print("PASS [3/6] Review confirm-all + collection audit + missing-class reporting")

        # 4. Verify participant-grouped split refuses incomplete 36 classes, and train_webcam_svm refuses without full 36 classes
        with (audit_dir / "collection_metadata.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assignment = auto_assign_participants(rows, seed=42)
        full_errors = validate_split(rows, assignment)[0]
        assert any("missing class Z" in e for e in full_errors)

        # Build subset split ONLY for testing training pipeline mechanics
        subset_errors = validate_split(rows, assignment, required_labels=classes)[0]
        assert subset_errors == [], subset_errors
        split_csv = root / "subset_split.csv"
        for r in rows:
            r["split"] = assignment[r["participant_id"]]
        with split_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        # Must REFUSE when --allow-subset-for-tests is NOT set!
        refused = False
        try:
            train_hog_main(
                [
                    "--split-csv", str(split_csv),
                    "--samples-dir", str(samples_dir),
                    "--model-output", str(root / "refused.joblib"),
                    "--report-dir", str(root / "reports_refused"),
                ]
            )
        except ValueError as exc:
            refused = "Training refused" in str(exc)
        assert refused, "train_webcam_svm MUST refuse incomplete 36-class data by default!"

        # Now run with --allow-subset-for-tests strictly in temp dir to verify end-to-end SVM training & baseline comparison
        out_model = root / "temp_webcam_hog_svm.joblib"
        out_reports = root / "temp_reports"
        rc = train_hog_main(
            [
                "--split-csv", str(split_csv),
                "--samples-dir", str(samples_dir),
                "--preprocess-modes", "legacy", "method_3b",
                "--c-grid", "0.01", "1.0",
                "--model-output", str(out_model),
                "--report-dir", str(out_reports),
                "--allow-subset-for-tests",
            ]
        )
        assert rc == 0 and out_model.is_file()
        bundle = joblib.load(out_model)
        assert bundle["model_family"] == "LinearSVC"
        assert bundle["feature_count"] == 1764
        assert bundle["is_full_36_class_model"] is False
        reloaded = SVMSignPredictor(model_path=out_model)
        test_pred = reloaded.predict_image_bgr(make_synthetic_pattern_bgr(999, 0))
        assert test_pred["predicted_label"] in classes
        print("PASS [4/6] Split refusal + LinearSVC validation selection + separate model + SVMSignPredictor reload")

        # 5. Verify MediaPipe landmark normalization & strict separation from HOG model
        fake_lms = [SimpleNamespace(x=0.1 + 0.01 * i, y=0.2 + 0.02 * i) for i in range(21)]
        vec = normalize_hand_landmarks_xy(fake_lms)
        assert vec.shape == (LANDMARK_FEATURE_DIM,) and np.isclose(vec[0], 0.0) and np.isclose(vec[1], 0.0)
        rejected_hog = False
        try:
            LandmarkSVMPredictor(DEFAULT_MODEL_PATH)
        except ValueError as exc:
            rejected_hog = "NOT a MediaPipe landmark SVM model" in str(exc)
        assert rejected_hog, "LandmarkSVMPredictor must reject 1764-dim HOG models!"
        # Verify Landmark SVM trainer also refuses incomplete 36-class dataset by default
        lm_refused = False
        try:
            train_landmark_main(
                [
                    "--split-csv", str(split_csv),
                    "--samples-dir", str(samples_dir),
                    "--model-output", str(root / "lm_refused.joblib"),
                    "--report-dir", str(root / "lm_reports_refused"),
                ]
            )
        except ValueError as exc:
            lm_refused = "Training refused" in str(exc)
        assert lm_refused, "train_landmark_svm_experiment MUST refuse incomplete 36-class data by default!"
        print("PASS [5/6] Landmark 42-dim normalization, refusal gate & strict HOG/Landmark model isolation")

        # 6. Verify 1-person pilot dry-run split (grouped by gesture_instance_id, zero leakage, refused by production trainer)
        pilot_dir = root / "pilot_samples"
        for c_idx, lbl in enumerate(["A", "B"]):
            for inst in range(1, 4):
                # Two distinct frames in the same gesture_instance_id -> must stay in the same split
                for frame_k in range(2):
                    img = make_synthetic_pattern_bgr(500 + c_idx * 100 + inst * 10 + frame_k, c_idx)
                    save_sample(
                        pilot_dir,
                        img,
                        {
                            "participant_id": "P01",
                            "session_id": "S01",
                            "gesture_instance_id": f"{lbl}_G0{inst}",
                            "handedness": "right",
                            "background": "plain_wall",
                            "lighting": "daylight",
                            "intended_sign": lbl,
                        },
                        roi_mode="manual",
                    )
        manage_main(["confirm-all", "--diag-dir", str(pilot_dir)])
        pilot_audit = root / "pilot_audit"
        export_collection_audit(load_saved_samples(pilot_dir), pilot_audit)
        pilot_split_out = root / "pilot_split_out"
        rc_pilot = split_main(
            [
                "--metadata", str(pilot_audit / "collection_metadata.csv"),
                "--pilot-instance-dry-run",
                "--output", str(pilot_split_out),
            ]
        )
        assert rc_pilot == 0 and (pilot_split_out / "pilot_split_dry_run.csv").is_file()
        pilot_readiness = json.loads((pilot_split_out / "readiness.json").read_text(encoding="utf-8"))
        assert pilot_readiness["pilot_only"] is True
        assert pilot_readiness["pilot_summary"]["zero_gesture_instance_leakage"] is True
        assert pilot_readiness["pilot_summary"]["zero_sha256_leakage"] is True
        # Production trainer MUST refuse pilot_split_dry_run.csv
        pilot_train_refused = False
        try:
            train_hog_main(
                [
                    "--split-csv", str(pilot_split_out / "pilot_split_dry_run.csv"),
                    "--samples-dir", str(pilot_dir),
                ]
            )
        except ValueError as exc:
            pilot_train_refused = "pilot_only=True" in str(exc) or "Training refused" in str(exc)
        assert pilot_train_refused, "Production trainer must refuse pilot_only split!"
        print("PASS [6/6] Pilot 1-person gesture-instance dry-run split (0 leakage) + production training refusal")


if __name__ == "__main__":
    run_all_checks()
