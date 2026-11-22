"""Ghi mẫu ký hiệu của chính bạn qua webcam để train thêm.

Chỉ lưu toạ độ 21 điểm khớp tay (không lưu ảnh) vào collected/samples.jsonl.
Tải về dạng .npz cùng định dạng với landmarks_raw.npz của notebook Kaggle, để gộp vào khi train.

Tắt tính năng này khi deploy công khai: đặt biến môi trường ENABLE_COLLECT=0.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from .features import MAX_SIDE, featurize, load_image_rgb

HERE = Path(__file__).resolve().parent
COLLECT_DIR = Path(os.getenv("COLLECT_DIR", "collected"))
SAMPLES = COLLECT_DIR / "samples.jsonl"
_lock = threading.Lock()


def _read_all() -> list[dict]:
    if not SAMPLES.exists():
        return []
    with open(SAMPLES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _counts() -> dict[str, int]:
    return dict(Counter(s["label"] for s in _read_all()))


def _paths(samples: list[dict]) -> list[str]:
    idx = Counter()
    out = []
    for s in samples:
        idx[(s["person"], s["label"])] += 1
        out.append(f"P{s['person']}_{s['label']}_{idx[(s['person'], s['label'])]:05d}.jpg")
    return out


def _npz_bytes(samples: list[dict]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf,
                        hands=np.array([s["hands"] for s in samples], np.float32),
                        mask=np.array([s["mask"] for s in samples], np.float32),
                        labels=np.array([s["label"] for s in samples]),
                        paths=np.array(_paths(samples)),
                        size=np.array([[s["w"], s["h"]] for s in samples], np.int32),
                        errors=np.array([], dtype=str))
    return buf.getvalue()


def _csv_bytes(samples: list[dict]) -> bytes:
    """Mỗi dòng 1 mẫu: thông tin + 2 tay x 21 điểm x (x, y, z) theo pixel."""
    cols = ["path", "label", "person", "w", "h", "pred", "mask0", "mask1"]
    cols += [f"h{t}_{i}_{c}" for t in range(2) for i in range(21) for c in "xyz"]
    lines = [",".join(cols)]
    for path, s in zip(_paths(samples), samples):
        flat = np.asarray(s["hands"], np.float32).reshape(-1)
        row = [path, s["label"], str(s["person"]), str(s["w"]), str(s["h"]), str(s.get("pred") or ""),
               *(str(int(m)) for m in s["mask"]), *(f"{v:.3f}" for v in flat)]
        lines.append(",".join(row))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _dataset_files(samples: list[dict]) -> dict[str, bytes]:
    cnt = Counter(s["label"] for s in samples)
    info = {"total": len(samples), "counts": dict(sorted(cnt.items())),
            "people": sorted({s["person"] for s in samples}),
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    readme = (
        "Tập mẫu ký hiệu ASL ghi qua trang /collect (chỉ toạ độ điểm khớp tay, không có ảnh).\n\n"
        "- collected_landmarks.npz : cùng định dạng landmarks_raw.npz (hands, mask, labels, paths, size, errors)\n"
        "- samples.jsonl           : dữ liệu gốc, mỗi dòng 1 mẫu\n"
        "- landmarks.csv           : bảng phẳng, toạ độ theo pixel (h<tay>_<điểm>_<x|y|z>)\n"
        "- stats.json              : số mẫu theo lớp và mã người\n")
    return {
        "collected_landmarks.npz": _npz_bytes(samples),
        "samples.jsonl": "".join(json.dumps(s) + "\n" for s in samples).encode("utf-8"),
        "landmarks.csv": _csv_bytes(samples),
        "stats.json": json.dumps(info, ensure_ascii=False, indent=2).encode("utf-8"),
        "README.txt": readme.encode("utf-8"),
    }


def make_router(state: dict) -> APIRouter:
    r = APIRouter(tags=["ghi mẫu"])

    @r.get("/collect", include_in_schema=False)
    def collect_page():
        return FileResponse(HERE / "static" / "collect.html")

    @r.post("/collect/sample")
    def add_sample(file: UploadFile = File(...), label: str = Form(...), person: int = Form(101)):
        """Nhận 1 khung hình + nhãn. Có tay thì lưu điểm khớp, trả về model hiện tại đoán gì."""
        clf = state["clf"]
        label = label.strip().upper()
        if label not in clf.labels:
            raise HTTPException(422, f"Nhãn '{label}' không có trong {clf.labels}")
        if not 100 <= person <= 9999:
            raise HTTPException(422, "Mã người nên từ 100 đến 9999 (1-10 là người của ASL-HG)")
        try:
            rgb = load_image_rgb(file.file.read(), MAX_SIDE)
        except Exception:
            raise HTTPException(400, "Không đọc được ảnh")
        hands, mask = state["detector"].detect(rgb)
        n = int(mask.sum())
        out = {"saved": False, "num_hands": n, "pred": None, "conf": None, "landmarks": None}
        if n == 0:
            return out
        h, w = rgb.shape[:2]
        top = clf.top_k(featurize(hands, mask), k=1)[0]
        out.update(pred=top["label"], conf=round(top["prob"], 4),
                   landmarks=(hands[:n] / np.array([w, h, w], np.float32)).round(4).tolist())
        rec = {"label": label, "person": person, "hands": hands.round(3).tolist(), "mask": mask.tolist(),
               "w": w, "h": h, "pred": top["label"], "ts": time.time()}
        with _lock:
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            with open(SAMPLES, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        out["saved"] = True
        return out

    @r.get("/collect/stats")
    def stats():
        samples = _read_all()
        cnt = Counter(s["label"] for s in samples)
        wrong = Counter(s["label"] for s in samples if s.get("pred") != s["label"])
        return {"total": len(samples), "counts": dict(cnt), "wrong_when_recorded": dict(wrong),
                "people": sorted({s["person"] for s in samples})}

    @r.post("/collect/clear")
    def clear(label: str = Form(...), person: int | None = Form(None)):
        """Xoá mẫu của một lớp (và một người nếu có) để ghi lại."""
        label = label.strip().upper()
        with _lock:
            samples = _read_all()
            keep = [s for s in samples if not (s["label"] == label and (person is None or s["person"] == person))]
            COLLECT_DIR.mkdir(parents=True, exist_ok=True)
            with open(SAMPLES, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(s) + "\n" for s in keep)
        return {"removed": len(samples) - len(keep), "counts": _counts()}

    @r.get("/collect/download")
    def download():
        """Tải toàn bộ mẫu dạng .npz (giống landmarks_raw.npz) để đưa lên Kaggle."""
        samples = _read_all()
        if not samples:
            raise HTTPException(404, "Chưa có mẫu nào")
        return Response(_npz_bytes(samples), media_type="application/octet-stream",
                        headers={"Content-Disposition": 'attachment; filename="collected_landmarks.npz"'})

    @r.get("/collect/download/zip")
    def download_zip():
        """Tải cả tập dataset trong 1 folder (nén .zip): npz + jsonl gốc + CSV + thống kê."""
        samples = _read_all()
        if not samples:
            raise HTTPException(404, "Chưa có mẫu nào")
        folder = "asl_dataset_" + time.strftime("%Y%m%d_%H%M%S")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in _dataset_files(samples).items():
                z.writestr(f"{folder}/{name}", data)
        return Response(buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{folder}.zip"'})

    @r.get("/collect/download/files")
    def download_files():
        """Các file của dataset (base64) để trang /collect ghi thẳng vào thư mục người dùng chọn."""
        samples = _read_all()
        if not samples:
            raise HTTPException(404, "Chưa có mẫu nào")
        return {"folder": "asl_dataset_" + time.strftime("%Y%m%d_%H%M%S"),
                "files": {k: base64.b64encode(v).decode() for k, v in _dataset_files(samples).items()}}

    return r
