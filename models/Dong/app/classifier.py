"""Chạy MLP đã train bằng numpy thuần: server không cần cài TensorFlow (~500 MB)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_DIM, FEATURE_VERSION


class MLPClassifier:
    def __init__(self, weights_path: str | Path, meta_path: str | Path):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if meta.get("feature_version") != FEATURE_VERSION:
            raise RuntimeError(
                f"Model train với feature_version={meta.get('feature_version')}, "
                f"code hiện tại là {FEATURE_VERSION}. Hãy train lại hoặc dùng đúng features.py."
            )
        self.meta = meta
        self.labels: list[str] = meta["labels"]

        w = np.load(weights_path)
        n = int(w["n_layers"])
        self.layers = [(w[f"W{i}"].astype(np.float32), w[f"b{i}"].astype(np.float32)) for i in range(n)]
        if self.layers[0][0].shape[0] != FEATURE_DIM:
            raise RuntimeError(f"Model nhận {self.layers[0][0].shape[0]} đặc trưng, code tạo {FEATURE_DIM}")
        if self.layers[-1][0].shape[1] != len(self.labels):
            raise RuntimeError("Số lớp đầu ra khác số nhãn trong model_meta.json")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        h = np.atleast_2d(x).astype(np.float32)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                np.maximum(h, 0, out=h)                      # ReLU
        h -= h.max(axis=1, keepdims=True)                     # softmax ổn định số
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    def top_k(self, x: np.ndarray, k: int = 5, mode: str = "all") -> list[dict]:
        """mode: "all" (36 lớp), "letters" (chỉ A-Z), "digits" (chỉ 0-9).
        Tách chế độ giúp tránh nhầm các cặp gần giống nhau như 2/V, 6/W, 1/Z, 0/O."""
        p = self.predict_proba(x)[0]
        if mode != "all":
            keep = np.array([lab.isdigit() == (mode == "digits") for lab in self.labels])
            p = np.where(keep, p, 0.0)
            p = p / max(p.sum(), 1e-12)                       # chuẩn hoá lại trong nhóm được chọn
        idx = [i for i in np.argsort(p)[::-1] if p[i] > 0][:k]
        return [{"label": self.labels[i], "prob": float(p[i])} for i in idx]
