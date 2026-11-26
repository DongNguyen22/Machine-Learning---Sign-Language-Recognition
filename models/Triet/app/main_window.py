"""Ứng dụng webcam ASL theo thời gian thực."""

from __future__ import annotations

import queue
import threading
import sys
from pathlib import Path
from typing import Any

import cv2
import customtkinter as ctk
import joblib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_extraction.geometric_features import FeatureExtractor  # noqa: E402
from src.inference.predictor import GesturePredictor  # noqa: E402
from src.preprocessing.coordinate_normalizer import Normalizer  # noqa: E402
from src.preprocessing.hand_detector import HandDetector  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ASLWindow(ctk.CTk):
    def __init__(self, camera_id: int = 0) -> None:
        super().__init__()
        self.title("ASL Alphabet Recognition")
        self.geometry("1000x720")
        self.camera_id = camera_id
        self.running = False
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.config = load_config()

        self.video = ctk.CTkLabel(self, text="")
        self.video.pack(fill="both", expand=True, padx=16, pady=16)
        self.result = ctk.CTkLabel(self, text="Đang khởi động...", font=("Arial", 24))
        self.result.pack(pady=8)

        self.detector = HandDetector(
            static_image_mode=False,
            max_num_hands=self.config.mediapipe.max_num_hands,
            model_complexity=self.config.mediapipe.model_complexity,
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
        )
        self.normalizer = Normalizer(self.config.features.scale_epsilon)
        self.extractor = FeatureExtractor()
        self.predictor = GesturePredictor(
            LABELS,
            ema_alpha=self.config.postprocess.ema_alpha,
            conf_threshold=self.config.postprocess.conf_threshold,
            margin_threshold=self.config.postprocess.margin_threshold,
            window_size=self.config.postprocess.window_size,
            min_votes=self.config.postprocess.min_votes,
            hold_seconds=self.config.postprocess.hold_seconds,
        )
        self.model = self._load_model()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _load_model(self) -> Any:
        production = ROOT / "models" / "production"
        candidates = [
            production / "exp3_mlp_opt.keras",
            production / "exp2_mlp_geo.keras",
            production / "exp1_mlp_base.keras",
            production / "exp3_svm_opt.joblib",
            production / "exp1_knn.joblib",
        ]
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            raise SystemExit("Thiếu mô hình trong models/production.")
        if selected.suffix == ".joblib":
            return joblib.load(selected)

        from tensorflow import keras
        return keras.models.load_model(selected)

    def camera_loop(self) -> None:
        # Camera and MediaPipe stay off the Tkinter thread; Tkinter only consumes frames through a bounded queue.
        capture = cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            self.after(0, lambda: self.result.configure(
                text=f"Không mở được camera {self.camera_id}"
            ))
            return

        try:
            while self.running:
                ok, frame = capture.read()
                if not ok:
                    continue

                label = None
                landmarks = self.detector.process(frame)
                if landmarks is not None:
                    normalized = self.normalizer.transform(landmarks)
                    features = self.extractor.transform(normalized)
                    x = features.reshape(1, -1)
                    if hasattr(self.model, "predict_proba"):
                        probabilities = np.asarray(
                            self.model.predict_proba(x)[0],
                            dtype=np.float32,
                        )
                    else:
                        probabilities = np.asarray(
                            self.model.predict(x, verbose=0)[0],
                            dtype=np.float32,
                        )
                    label = self.predictor.update(probabilities)

                if label:
                    self.after(0, lambda text=label: self.result.configure(
                        text=f"Ký hiệu: {text}"
                    ))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = ctk.CTkImage(
                    light_image=Image.fromarray(rgb),
                    size=(960, 620),
                )
                try:
                    self.frames.put_nowait(image)
                except queue.Full:
                    try:
                        self.frames.get_nowait()
                        self.frames.put_nowait(image)
                    except queue.Empty:
                        pass
        finally:
            capture.release()

    def update_frame(self) -> None:
        if not self.running:
            return
        try:
            image = self.frames.get_nowait()
            self.video.configure(image=image)
        except queue.Empty:
            pass
        self.after(30, self.update_frame)

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self.camera_loop, daemon=True).start()
        self.update_frame()

    def close(self) -> None:
        self.running = False
        self.detector.close()
        super().destroy()


def main() -> None:
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    window = ASLWindow(camera_id=0)
    window.start()
    window.mainloop()


if __name__ == "__main__":
    main()
