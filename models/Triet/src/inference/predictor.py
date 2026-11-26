"""Bộ lọc dự đoán ký hiệu cho luồng webcam."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


class GesturePredictor:
    """Tích hợp EMA, kiểm tra chênh lệch xác suất và cửa sổ bỏ phiếu."""

    def __init__(
        self,
        labels: Sequence[str],
        ema_alpha: float = 0.4,
        conf_threshold: float = 0.75,
        margin_threshold: float = 0.20,
        window_size: int = 10,
        min_votes: int = 7,
        hold_seconds: float = 1.2,
    ) -> None:
        self.labels = list(labels)
        self.ema_alpha = ema_alpha
        self.conf_threshold = conf_threshold
        self.margin_threshold = margin_threshold
        self.window = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.hold_seconds = hold_seconds
        self.ema: np.ndarray | None = None

    def update(self, probabilities: np.ndarray) -> str | None:
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if values.shape[0] != len(self.labels) or not np.isfinite(values).all():
            raise ValueError("Xác suất phải khớp 26 nhãn và không chứa NaN/Inf.")

        if self.ema is None:
            self.ema = values.copy()
        else:
            self.ema = (
                self.ema_alpha * values
                + (1.0 - self.ema_alpha) * self.ema
            )

        order = np.argsort(self.ema)[::-1]
        label_index = int(order[0])
        confidence = float(self.ema[label_index])
        margin = confidence - float(self.ema[int(order[1])])
        if confidence < self.conf_threshold or margin < self.margin_threshold:
            return None

        label = self.labels[label_index]
        self.window.append(label)
        if self.window.count(label) >= self.min_votes:
            return label
        return None

    def reset(self) -> None:
        self.ema = None
        self.window.clear()
