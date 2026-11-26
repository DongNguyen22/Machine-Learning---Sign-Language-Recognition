"""Automated HTTP integration tests for multi-scope Landmark SVM (/predict/landmark)
across `all_36`, `letters_az`, `digits_09` and backward compatibility of default HOG (/predict, /health).
"""

from __future__ import annotations

import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import cv2
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict_landmark_v2 import (
    DEFAULT_LANDMARK_09_MODEL_PATH,
    DEFAULT_LANDMARK_AZ_MODEL_PATH,
    DEFAULT_LANDMARK_V2_MODEL_PATH,
    VALID_LANDMARK_SCOPES,
    compute_ovr_heuristic_confidence,
)
from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.model.svm_api_server import DEFAULT_ALLOWED_ORIGINS, SVMAPIApp, create_request_handler
from src.preprocess.webcam_features import (
    LANDMARK_GEOM_V2_FEATURE_DIM,
    LANDMARK_GEOM_V2_FEATURE_SCHEMA,
    canonicalize_saved_crop_bgr,
    create_hand_landmarker,
    extract_landmark_geometric_features_v2,
    normalize_hand_landmarks_geom_v2,
)

PILOT_DIR = PROJECT_ROOT / "data" / "webcam_pilot"
CACHE_PATH = PROJECT_ROOT / "results" / "svm" / "experiments" / "landmark_asl_v1" / "asl_landmarks_cache.npz"
LETTERS_SET = {chr(c) for c in range(ord("A"), ord("Z") + 1)}
DIGITS_SET = {str(i) for i in range(10)}


def http_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read().decode("utf-8"))


class TestLandmarkV2API(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = SVMAPIApp(
            model_path=PROJECT_ROOT / DEFAULT_MODEL_PATH,
            allowed_origins=DEFAULT_ALLOWED_ORIGINS,
            landmark_v2_model_path=DEFAULT_LANDMARK_V2_MODEL_PATH,
            landmark_az_model_path=DEFAULT_LANDMARK_AZ_MODEL_PATH,
            landmark_09_model_path=DEFAULT_LANDMARK_09_MODEL_PATH,
        )
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), create_request_handler(cls.app))
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_01_health_reports_all_3_scopes_and_preserves_hog_fields(self) -> None:
        status, body = http_json(f"{self.base_url}/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["model_ready"])
        self.assertEqual(body["feature_count"], 1764)
        self.assertEqual(body["class_count"], 36)
        self.assertEqual(len(body["labels"]), 36)

        self.assertTrue(body["landmark_v2_available"])
        lm = body["landmark_v2"]
        self.assertTrue(lm["available"])
        self.assertEqual(lm["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
        self.assertEqual(lm["feature_dim"], LANDMARK_GEOM_V2_FEATURE_DIM)
        self.assertEqual(lm["class_count"], 36)

        scopes = lm["scopes"]
        self.assertEqual(set(scopes.keys()), set(VALID_LANDMARK_SCOPES))
        self.assertTrue(scopes["all_36"]["available"])
        self.assertEqual(scopes["all_36"]["class_count"], 36)
        self.assertTrue(scopes["letters_az"]["available"])
        self.assertEqual(scopes["letters_az"]["class_count"], 26)
        self.assertEqual(set(scopes["letters_az"]["labels"]), LETTERS_SET)
        self.assertTrue(scopes["digits_09"]["available"])
        self.assertEqual(scopes["digits_09"]["class_count"], 10)
        self.assertEqual(set(scopes["digits_09"]["labels"]), DIGITS_SET)

    def test_02_predict_landmark_3_scopes_bitwise_parity_and_confidence(self) -> None:
        import mediapipe as mp

        scope_models = {
            "all_36": joblib.load(DEFAULT_LANDMARK_V2_MODEL_PATH),
            "letters_az": joblib.load(DEFAULT_LANDMARK_AZ_MODEL_PATH),
            "digits_09": joblib.load(DEFAULT_LANDMARK_09_MODEL_PATH),
        }

        # Extract real MediaPipe landmarks from 4 confirmed pilot images (A, B, C, D)
        detector = create_hand_landmarker(min_detection_confidence=0.3, num_hands=1)
        picked: dict[str, tuple[list[dict[str, float]], int, int]] = {}
        try:
            for sdir in sorted(PILOT_DIR.iterdir()):
                if not sdir.is_dir():
                    continue
                meta = json.loads((sdir / "diagnostic_metadata.json").read_text(encoding="utf-8"))
                lbl = str(meta.get("verified_true_label", ""))
                if lbl in picked or lbl not in {"A", "B", "C", "D"}:
                    continue
                bgr = cv2.imread(str(sdir / "received_crop.png"))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                pad = max(16, int(round(max(h, w) * 0.5)))
                rgb_pad = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(232, 232, 232))
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_pad))
                if not res.hand_landmarks:
                    continue
                in_h, in_w = rgb_pad.shape[:2]
                pts = [{"x": float(p.x), "y": float(p.y)} for p in res.hand_landmarks[0]]
                picked[lbl] = (pts, in_w, in_h)
                if len(picked) == 4:
                    break
        finally:
            detector.close()

        self.assertEqual(len(picked), 4)

        latencies_by_scope: dict[str, list[float]] = {sc: [] for sc in VALID_LANDMARK_SCOPES}
        for scope in VALID_LANDMARK_SCOPES:
            bundle = scope_models[scope]
            direct_pipe = bundle["model"]
            class_names = [str(c) for c in bundle["class_names"]]
            clf_classes = np.asarray(direct_pipe.named_steps["clf"].classes_, dtype=np.int64)

            for lbl, (pts, w, h) in picked.items():
                feat_86 = normalize_hand_landmarks_geom_v2(pts, image_width=w, image_height=h)
                X = feat_86.reshape(1, -1)
                direct_pred_idx = int(direct_pipe.predict(X)[0])
                direct_pred_label = class_names[direct_pred_idx]
                direct_scores = np.asarray(direct_pipe.decision_function(X)[0], dtype=np.float64)

                req_payload: dict[str, object] = {
                    "landmarks": pts,
                    "image_width": w,
                    "image_height": h,
                    "top_k": 3,
                    "frame_id": f"{scope}_{lbl}",
                    "scope": scope,
                }
                t0 = time.perf_counter()
                status, body = http_json(f"{self.base_url}/predict/landmark", req_payload)
                latencies_by_scope[scope].append((time.perf_counter() - t0) * 1000.0)

                self.assertEqual(status, 200)
                self.assertEqual(body["scope"], scope)
                self.assertEqual(body["feature_schema"], LANDMARK_GEOM_V2_FEATURE_SCHEMA)
                self.assertEqual(body["feature_count"], LANDMARK_GEOM_V2_FEATURE_DIM)
                self.assertEqual(body["predicted_label"], direct_pred_label)
                if scope == "letters_az":
                    self.assertIn(body["predicted_label"], LETTERS_SET)
                elif scope == "digits_09":
                    self.assertIn(body["predicted_label"], DIGITS_SET)

                top3 = body["top_k"]
                self.assertEqual(len(top3), 3)
                for item in top3:
                    cid = int(item["class_index"])
                    pos = int(np.where(clf_classes == cid)[0][0])
                    self.assertEqual(float(item["decision_score"]), float(direct_scores[pos]))
                    if scope == "letters_az":
                        self.assertIn(item["label"], LETTERS_SET)
                    elif scope == "digits_09":
                        self.assertIn(item["label"], DIGITS_SET)

                conf = float(body["confidence"])
                self.assertGreater(conf, 0.0)
                self.assertLessEqual(conf, 0.99)

        # Verify omitting `scope` defaults cleanly to `all_36`
        sample_pts, sw, sh = picked["A"]
        st_def, body_def = http_json(
            f"{self.base_url}/predict/landmark",
            {"landmarks": sample_pts, "image_width": sw, "image_height": sh, "top_k": 3},
        )
        st_exp, body_exp = http_json(
            f"{self.base_url}/predict/landmark",
            {"landmarks": sample_pts, "image_width": sw, "image_height": sh, "top_k": 3, "scope": "all_36"},
        )
        self.assertEqual(st_def, 200)
        self.assertEqual(body_def["scope"], "all_36")
        self.assertEqual(body_def["predicted_label"], body_exp["predicted_label"])
        self.assertEqual(body_def["predicted_decision_score"], body_exp["predicted_decision_score"])

        # Verify validation confidence distribution across all 3 scopes: 0.00% > 0.99 & good separation
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            val_mask = (z["new_split"] == "validation") & z["detected"]
            X_val_xy = np.asarray(z["X_landmarks"][val_mask], dtype=np.float32)
            y_val_36 = np.asarray(z["y"][val_mask], dtype=np.int32)
            cn36 = [str(c) for c in z["class_names"]]

        for scope in VALID_LANDMARK_SCOPES:
            bundle = scope_models[scope]
            pipe = bundle["model"]
            sub_cls = [str(c) for c in bundle["class_names"]]
            gids = [cn36.index(c) for c in sub_cls]
            g2l = {g: l for l, g in enumerate(gids)}
            m = np.isin(y_val_36, gids)
            Xv = extract_landmark_geometric_features_v2(X_val_xy[m])
            yv = np.array([g2l[int(v)] for v in y_val_36[m]], dtype=np.int32)
            yp = np.asarray(pipe.predict(Xv), dtype=np.int32)
            sc = np.asarray(pipe.decision_function(Xv), dtype=np.float64)
            clfs = np.asarray(pipe.named_steps["clf"].classes_, dtype=np.int64)
            confs = np.empty(len(yv), dtype=np.float64)
            for i in range(len(yv)):
                ord_i = np.argsort(sc[i])[::-1]
                ppos = int(np.where(clfs == yp[i])[0][0])
                if int(ord_i[0]) != ppos:
                    ord_i = np.concatenate([[ppos], ord_i[ord_i != ppos]])
                confs[i], _, _ = compute_ovr_heuristic_confidence(sc[i], ord_i[:3], len(sub_cls))
            ok = yp == yv
            sat_pct = float(np.mean(confs > 0.99) * 100.0)
            self.assertEqual(sat_pct, 0.0, f"Scope {scope} saturated >0.99: {sat_pct:.2f}%")
            self.assertGreater(float(confs[ok].mean()), float(confs[~ok].mean()) + 0.05)
            print(
                f"\n[Scope {scope:10s}] HTTP avg={np.mean(latencies_by_scope[scope]):5.2f} ms | "
                f"val conf ok={confs[ok].mean():.4f} vs err={confs[~ok].mean():.4f} | "
                f"range=[{confs.min():.4f}, {confs.max():.4f}] | >0.99={sat_pct:.2f}%",
                flush=True,
            )

    def test_03_invalid_landmark_inputs_and_invalid_scope_return_400(self) -> None:
        valid_21 = [{"x": 0.1 + i * 0.01, "y": 0.2 + i * 0.01} for i in range(21)]
        bad_cases = [
            {},
            {"landmarks": valid_21[:20], "image_width": 640, "image_height": 480},
            {"landmarks": valid_21 + [{"x": 0.5, "y": 0.5}], "image_width": 640, "image_height": 480},
            {"landmarks": [{"x": "bad", "y": 0.1}] + valid_21[1:], "image_width": 640, "image_height": 480},
            {"landmarks": valid_21, "image_width": 0, "image_height": 480},
            {"landmarks": valid_21, "image_width": 640, "image_height": -5},
            {"landmarks": valid_21, "image_width": 640, "image_height": 480, "top_k": 0},
            {"landmarks": valid_21, "image_width": 640, "image_height": 480, "scope": "invalid_scope"},
            {"landmarks": valid_21, "image_width": 640, "image_height": 480, "scope": 123},
        ]
        for case in bad_cases:
            st, body = http_json(f"{self.base_url}/predict/landmark", case)
            self.assertEqual(st, 400, f"Expected 400 for {case}, got {st}: {body}")
            self.assertIn("error", body)

        raw_nan_body = (
            '{"landmarks": ['
            + '{"x": NaN, "y": 0.2},'
            + ",".join('{"x": 0.2, "y": 0.3}' for _ in range(20))
            + '], "image_width": 640, "image_height": 480}'
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/predict/landmark",
            data=raw_nan_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 400)

    def test_04_missing_one_scope_isolated_503_while_other_scopes_and_hog_work(self) -> None:
        # Missing ONLY `digits_09`; `all_36`, `letters_az`, and HOG `/predict` MUST stay healthy!
        partial_app = SVMAPIApp(
            model_path=PROJECT_ROOT / DEFAULT_MODEL_PATH,
            allowed_origins=DEFAULT_ALLOWED_ORIGINS,
            landmark_v2_model_path=DEFAULT_LANDMARK_V2_MODEL_PATH,
            landmark_az_model_path=DEFAULT_LANDMARK_AZ_MODEL_PATH,
            landmark_09_model_path=PROJECT_ROOT / "models" / "svm" / "experiments" / "missing_09.joblib",
        )
        srv = ThreadingHTTPServer(("127.0.0.1", 0), create_request_handler(partial_app))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{srv.server_port}"
            st_h, body_h = http_json(f"{url}/health")
            self.assertEqual(st_h, 200)
            self.assertTrue(body_h["model_ready"])
            self.assertTrue(body_h["landmark_v2_available"])
            scopes = body_h["landmark_v2"]["scopes"]
            self.assertTrue(scopes["all_36"]["available"])
            self.assertTrue(scopes["letters_az"]["available"])
            self.assertFalse(scopes["digits_09"]["available"])

            valid_21 = [{"x": 0.1 + i * 0.01, "y": 0.2 + i * 0.01} for i in range(21)]
            st_09, body_09 = http_json(
                f"{url}/predict/landmark",
                {"landmarks": valid_21, "image_width": 640, "image_height": 480, "scope": "digits_09"},
            )
            self.assertEqual(st_09, 503)
            self.assertIn("error", body_09)

            st_az, body_az = http_json(
                f"{url}/predict/landmark",
                {"landmarks": valid_21, "image_width": 640, "image_height": 480, "scope": "letters_az"},
            )
            self.assertEqual(st_az, 200)
            self.assertIn(body_az["predicted_label"], LETTERS_SET)

            # Verify default HOG /predict works bit-for-bit identically to SVMSignPredictor
            sample_dir = sorted([d for d in PILOT_DIR.iterdir() if d.is_dir()])[0]
            crop_bytes = (sample_dir / "received_crop.png").read_bytes()
            crop_bgr = cv2.imread(str(sample_dir / "received_crop.png"))
            meta = json.loads((sample_dir / "diagnostic_metadata.json").read_text(encoding="utf-8"))
            direct_hog = SVMSignPredictor(PROJECT_ROOT / DEFAULT_MODEL_PATH).predict_image_bgr(
                canonicalize_saved_crop_bgr(crop_bgr, metadata=meta),
                top_k=3,
                experimental_webcam_mode=True,
                already_flipped=False,
            )
            st_p, body_p = http_json(
                f"{url}/predict",
                {
                    "image": base64.b64encode(crop_bytes).decode("ascii"),
                    "top_k": 3,
                    "experimental_webcam_mode": True,
                    "already_flipped": False,
                },
            )
            self.assertEqual(st_p, 200)
            self.assertEqual(body_p["predicted_label"], direct_hog["predicted_label"])
            self.assertEqual(
                float(body_p["predicted_decision_score"]),
                float(direct_hog["predicted_decision_score"]),
            )
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
