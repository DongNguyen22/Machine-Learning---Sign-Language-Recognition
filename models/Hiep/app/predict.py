"""Chạy dự đoán ASL bằng SA-ResMLP trên ảnh tĩnh hoặc webcam (có vẽ khớp tay)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .features import extract_181_features
from .model_def import SA_ResMLP

APP_DIR = Path(__file__).resolve().parent
CKPT_PATH = APP_DIR / "model" / "sa_resmlp_best.pt"
LABELS_PATH = APP_DIR / "model" / "labels.json"
TASK_PATH = APP_DIR / "model" / "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
]


def load_predictor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    labels: list[str] = (
        json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        if LABELS_PATH.exists()
        else ckpt["labels"]
    )
    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)

    model = SA_ResMLP(input_dim=int(ckpt["input_dim"]), num_classes=len(labels)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TASK_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    landmarker = vision.HandLandmarker.create_from_options(opts)
    return model, labels, mean, std, device, landmarker


def draw_landmarks(frame: np.ndarray, hand_landmarks_list) -> None:
    """Vẽ 21 khớp tay và các đường nối xương trực tiếp lên frame."""
    h, w = frame.shape[:2]
    for lms in hand_landmarks_list:
        pts = [(int(p.x * w), int(p.y * h)) for p in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (0, 128, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)


def predict_frame(bgr: np.ndarray, model, labels, mean, std, device, landmarker, top_k: int = 3):
    h, w = bgr.shape[:2]
    if max(h, w) > 640:
        s = 640 / max(h, w)
        bgr_in = cv2.resize(bgr, (int(w * s), int(h * s)))
        h, w = bgr_in.shape[:2]
    else:
        bgr_in = bgr

    rgb = cv2.cvtColor(bgr_in, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    vec, ok = extract_181_features(res, w, h)
    if not ok:
        return [], res.hand_landmarks

    x = (vec[None, :] - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x).to(device)), dim=-1)[0].cpu().numpy()
    idx = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in idx], res.hand_landmarks


def main():
    parser = argparse.ArgumentParser(description="SA-ResMLP ASL Predictor")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh cần test (bỏ trống để mở webcam)")
    args = parser.parse_args()

    model, labels, mean, std, device, landmarker = load_predictor()
    try:
        if args.image:
            bgr = cv2.imread(args.image)
            if bgr is None:
                raise FileNotFoundError(f"Không đọc được ảnh: {args.image}")
            preds, _ = predict_frame(bgr, model, labels, mean, std, device, landmarker)
            print("Kết quả:", preds if preds else "Không phát hiện bàn tay")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Không mở được webcam")
        print("Đang chạy webcam... Nhấn 'q' để thoát.")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            preds, hand_lms = predict_frame(frame, model, labels, mean, std, device, landmarker)
            if hand_lms:
                draw_landmarks(frame, hand_lms)
            text = f"{preds[0][0]} ({preds[0][1]:.1%})" if preds else "No hand"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
            cv2.imshow("SA-ResMLP ASL", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    finally:
        landmarker.close()


if __name__ == "__main__":
    main()
