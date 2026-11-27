"""MediaPipe Hands wrapper for static images and webcam streams."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr is None:
            return None
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] < 3:
            return None
        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0].landmark
        points = np.asarray(
            [(point.x, point.y, point.z) for point in landmarks],
            dtype=np.float32,
        )
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return None
        return points

    def detect(self, image_bgr: np.ndarray) -> np.ndarray | None:
        return self.process(image_bgr)

    def set_static_image_mode(self, enabled: bool) -> None:
        self.close()
        self.static_image_mode = enabled
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=enabled,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def close(self) -> None:
        if hasattr(self, "_hands") and self._hands is not None:
            self._hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
