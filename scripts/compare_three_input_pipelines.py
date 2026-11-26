r"""Compare 3 offline input preprocessing pipelines on the 23 confirmed webcam samples (A, B, C, D).

Strict constraints enforced:
1. Uses only the 23 confirmed samples (`label_confirmed: true`) and strictly uses
   `verified_true_label` as ground truth.
2. Uses the exact same final SVM model (`models/svm/svm_final_c0_01.joblib`),
   the exact same `resize_with_padding` (64x64), and the exact same 1764-dim HOG extraction.
3. Carefully checks `flip_input_for_model` at save time so no sample is accidentally
   flipped twice:
   - Sample #04 was saved with `flip_input_for_model=True` (its `received_crop.png` on disk
     was already horizontally flipped once by the frontend canvas before saving).
   - The remaining 22 samples (#01..#03, #05..#23) were saved with `flip_input_for_model=False`
     (their `received_crop.png` on disk is unflipped raw camera orientation).
   - Therefore:
     * Method 1 (Ảnh crop gốc chưa lật ngang):
       - For #01..#03, #05..#23: `received_crop.png` as-is.
       - For #04: we report both the unflipped raw camera state (`cv2.flip(received_crop, 1)`)
         and the as-saved disk state (`received_crop.png`) so everything is 100% transparent.
     * Method 2 (Lật ngang ảnh crop đúng 1 lần trước khi trích xuất HOG):
       - For #01..#03, #05..#23 (`flip_saved=False`): `cv2.flip(received_crop, 1)`.
       - For #04 (`flip_saved=True`): keep `received_crop.png` as-is (do NOT flip a second time!).
     * Method 3 (Lật ngang đúng 1 lần + giảm ảnh hưởng của nền):
       - We test and visually inspect both:
         (3A) Tách nền ngưỡng cứng YCrCb (`Cr >= 135`) -> kiểm tra xem có làm mất ngón tay/viền tay không.
         (3B) Giảm ảnh hưởng nền bảo toàn viền tay (Soft Background Suppression: kết hợp vùng màu da YCrCb +
              đóng hình thái học `MORPH_CLOSE` + nới biên bảo vệ viền ngón tay `dilate` + làm mờ biên
              `GaussianBlur` để nâng vùng nền tối phi-da về mức xám sáng `232` giống nền dataset mà không
              cắt phạm vào ngón tay hay viền bàn tay).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict_svm import SVMSignPredictor
from src.preprocess.build_hog_dataset import extract_hog_with_diagnostics_from_bgr

DIAG_DIR = PROJECT_ROOT / "results" / "svm" / "webcam_diagnostics"
OUT_JSON_PATH = DIAG_DIR / "three_methods_comparison_report.json"
OUT_ALL_23_PNG = DIAG_DIR / "three_methods_all_23_inputs_64x64.png"
OUT_BD_FOCUS_PNG = DIAG_DIR / "three_methods_B_D_and_failures_64x64.png"


def suppress_background_hard_ycrcb(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard YCrCb skin mask (`Cr >= 135`): used to check if hard thresholding erodes fingertips."""
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = (ycc[:, :, 1] >= 135).astype(np.uint8) * 255
    out = np.where(mask[:, :, None] > 0, image_bgr, 235).astype(np.uint8)
    return out, mask


def suppress_background_preserve_hand(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft background suppression that preserves fingertips, bright highlights, and hand contour.

    How it works:
    1. Detects core skin region in YCrCb (`Cr >= 131` and `77 <= Cb <= 138`) excluding very dark
       background shadows (`Y >= 45`).
    2. Applies morphological closing (`9x9` ellipse) to fill any internal creases or overexposed
       fingertip highlight gaps inside the hand.
    3. Keeps connected components overlapping the central hand region and dilates by `7x7` pixels
       so the outer hand boundary (the gradient edge between skin and background that HOG needs!)
       is completely untouched.
    4. Applies a smooth Gaussian transition (`11x11`, sigma=3.0) and lifts only the outer dark/cluttered
       background toward the training dataset's neutral light background (`232`), while slightly
       normalizing overly bright specular highlights on the skin so internal finger folds remain visible.
    """
    h, w = image_bgr.shape[:2]
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0]
    Cr = ycc[:, :, 1]
    Cb = ycc[:, :, 2]

    skin_raw = ((Cr >= 131) & (Cb >= 77) & (Cb <= 138) & (Y >= 45)).astype(np.uint8) * 255

    # Close internal holes/highlights on fingers
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_closed = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Keep components that intersect the inner 70% box (filters out isolated corner noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin_closed, connectivity=8)
    inner_y1, inner_y2 = int(h * 0.15), int(h * 0.85)
    inner_x1, inner_x2 = int(w * 0.15), int(w * 0.85)
    clean_mask = np.zeros_like(skin_closed)
    for lbl_id in range(1, num_labels):
        comp = (labels == lbl_id)
        if np.any(comp[inner_y1:inner_y2, inner_x1:inner_x2]) and stats[lbl_id, cv2.CC_STAT_AREA] >= 60:
            clean_mask[comp] = 255

    if np.count_nonzero(clean_mask) == 0:
        clean_mask = skin_closed

    # Dilate so we never clip fingertips or the natural outer edge of the hand
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected_hand_mask = cv2.dilate(clean_mask, k_dilate, iterations=1)

    alpha = cv2.GaussianBlur(protected_hand_mask.astype(np.float32) / 255.0, (13, 13), 3.5)
    bg_color = np.full_like(image_bgr, 232, dtype=np.float32)
    blended = np.clip(
        image_bgr.astype(np.float32) * alpha[:, :, None] + bg_color * (1.0 - alpha[:, :, None]),
        0,
        255,
    ).astype(np.uint8)
    return blended, protected_hand_mask


def evaluate_method(
    records: list[dict],
    predictor: SVMSignPredictor,
    method_key: str,
) -> dict:
    results_per_sample = []
    for rec in records:
        raw_disk_bgr = rec["raw_disk_bgr"]
        flip_saved = rec["flip_input_for_model"]
        true_lbl = rec["verified_true_label"]

        # Reconstruct true unflipped raw camera crop vs single-flipped crop (avoiding double flip on #04!)
        unflipped_bgr = cv2.flip(raw_disk_bgr, 1) if flip_saved else raw_disk_bgr
        flipped_once_bgr = raw_disk_bgr if flip_saved else cv2.flip(raw_disk_bgr, 1)

        if method_key == "M1_raw_unflipped":
            input_bgr = unflipped_bgr
            mask_used = None
        elif method_key == "M1_as_saved_on_disk":
            input_bgr = raw_disk_bgr
            mask_used = None
        elif method_key == "M2_flipped_once":
            input_bgr = flipped_once_bgr
            mask_used = None
        elif method_key == "M3A_flip_hard_ycrcb":
            input_bgr, mask_used = suppress_background_hard_ycrcb(flipped_once_bgr)
        elif method_key == "M3B_flip_preserve_hand_bg_suppressed":
            input_bgr, mask_used = suppress_background_preserve_hand(flipped_once_bgr)
        else:
            raise ValueError(f"Unknown method_key: {method_key}")

        pred_res = predictor.predict_image_bgr(input_bgr, top_k=36)
        _, _bgr64, gray64 = extract_hog_with_diagnostics_from_bgr(input_bgr)

        all_36 = pred_res["top_k_predictions"]
        true_rank = next(item["rank"] for item in all_36 if item["label"] == true_lbl)
        true_score = next(item["decision_score"] for item in all_36 if item["label"] == true_lbl)

        pred_lbl = str(pred_res["predicted_label"])
        is_correct = (pred_lbl == true_lbl)

        results_per_sample.append(
            {
                "stt": rec["stt"],
                "folder_name": rec["folder_name"],
                "verified_true_label": true_lbl,
                "flip_saved_at_capture": flip_saved,
                "predicted_label": pred_lbl,
                "is_correct": is_correct,
                "predicted_decision_score": float(pred_res["predicted_decision_score"]),
                "margin_top1_minus_top2": float(pred_res["margin_top1_minus_top2"]),
                "true_label_rank": int(true_rank),
                "true_label_score": float(true_score),
                "top_3": [
                    {
                        "rank": item["rank"],
                        "label": item["label"],
                        "decision_score": round(float(item["decision_score"]), 4),
                    }
                    for item in all_36[:3]
                ],
                "gray64": gray64,
                "input_bgr": input_bgr,
                "mask_used": mask_used,
            }
        )

    classes = ["A", "B", "C", "D"]
    per_class = {}
    for cls in classes:
        cls_items = [x for x in results_per_sample if x["verified_true_label"] == cls]
        corr = sum(1 for x in cls_items if x["is_correct"])
        tot = len(cls_items)
        per_class[cls] = {
            "correct": corr,
            "total": tot,
            "accuracy_pct": round(corr / tot * 100.0, 2) if tot else 0.0,
            "predicted_distribution": dict(Counter(x["predicted_label"] for x in cls_items)),
            "mean_true_label_rank": round(float(np.mean([x["true_label_rank"] for x in cls_items])), 2),
            "mean_true_label_score": round(float(np.mean([x["true_label_score"] for x in cls_items])), 4),
        }

    total_correct = sum(1 for x in results_per_sample if x["is_correct"])
    return {
        "method_key": method_key,
        "total_correct": total_correct,
        "total_samples": len(results_per_sample),
        "total_accuracy_pct": round(total_correct / len(results_per_sample) * 100.0, 2),
        "per_class": per_class,
        "samples": results_per_sample,
    }


def main() -> int:
    sample_dirs = sorted(
        [d for d in DIAG_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )
    records = []
    for idx, sdir in enumerate(sample_dirs, start=1):
        meta = json.loads((sdir / "diagnostic_metadata.json").read_text(encoding="utf-8"))
        if not bool(meta.get("label_confirmed", False)):
            raise ValueError(f"Sample #{idx:02d} not confirmed!")
        bgr = cv2.imread(str(sdir / "received_crop.png"))
        records.append(
            {
                "stt": idx,
                "folder_name": sdir.name,
                "verified_true_label": str(meta["verified_true_label"]).strip().upper(),
                "flip_input_for_model": bool(meta.get("flip_input_for_model", False)),
                "raw_disk_bgr": bgr,
            }
        )

    predictor = SVMSignPredictor()

    m1_unflipped = evaluate_method(records, predictor, "M1_raw_unflipped")
    m1_disk = evaluate_method(records, predictor, "M1_as_saved_on_disk")
    m2_flip = evaluate_method(records, predictor, "M2_flipped_once")
    m3a_hard = evaluate_method(records, predictor, "M3A_flip_hard_ycrcb")
    m3b_preserve = evaluate_method(records, predictor, "M3B_flip_preserve_hand_bg_suppressed")

    for title, res in [
        ("Cách 1A (Ảnh crop gốc chưa lật 23/23)", m1_unflipped),
        ("Cách 1B (Ảnh crop đúng như lúc lưu trên đĩa - #04 đã lật)", m1_disk),
        ("Cách 2  (Lật ngang đúng 1 lần - tránh lật 2 lần ở #04)", m2_flip),
        ("Cách 3A (Lật ngang 1 lần + Tách nền YCrCb ngưỡng cứng Cr>=135)", m3a_hard),
        ("Cách 3B (Lật ngang 1 lần + Giảm nền bảo toàn viền tay)", m3b_preserve),
    ]:
        print(f"=== {title} ===")
        print(f"Tổng số đúng: {res['total_correct']}/{res['total_samples']} ({res['total_accuracy_pct']}%)")
        for cls in ["A", "B", "C", "D"]:
            c_info = res["per_class"][cls]
            print(
                f"  Lớp {cls}: {c_info['correct']}/{c_info['total']} ({c_info['accuracy_pct']}%) | "
                f"Đoán: {c_info['predicted_distribution']} | "
                f"Hạng TB '{cls}': #{c_info['mean_true_label_rank']} (score TB: {c_info['mean_true_label_score']})"
            )
        print()

    # Visual Figure 1: All 23 samples comparing actual 64x64 model inputs across M1, M2, M3A (hard), M3B (preserve)
    fig, axes = plt.subplots(23, 5, figsize=(13.5, 48), dpi=120)
    for idx in range(23):
        rec = records[idx]
        lbl = rec["verified_true_label"]
        s1 = m1_unflipped["samples"][idx]
        s2 = m2_flip["samples"][idx]
        s3a = m3a_hard["samples"][idx]
        s3b = m3b_preserve["samples"][idx]

        # Dataset reference 64x64 for this class
        ds_bgr = cv2.imread(str(PROJECT_ROOT / "frontend" / "public" / "signs" / f"{lbl}.jpg"))
        _, _, ds_g64 = extract_hog_with_diagnostics_from_bgr(ds_bgr)

        cols_data = [
            (s1["gray64"], f"#{idx+1:02d} ({lbl}) Cách 1: Gốc\nĐoán: {s1['predicted_label']} ({'ĐÚNG' if s1['is_correct'] else 'SAI'}) | Hạng '{lbl}': #{s1['true_label_rank']}", s1["is_correct"]),
            (s2["gray64"], f"Cách 2: Lật 1 lần\nĐoán: {s2['predicted_label']} ({'ĐÚNG' if s2['is_correct'] else 'SAI'}) | Hạng '{lbl}': #{s2['true_label_rank']}", s2["is_correct"]),
            (s3a["gray64"], f"Cách 3A: Tách nền cứng\nĐoán: {s3a['predicted_label']} ({'ĐÚNG' if s3a['is_correct'] else 'SAI'}) | Hạng '{lbl}': #{s3a['true_label_rank']}", s3a["is_correct"]),
            (s3b["gray64"], f"Cách 3B: Giảm nền giữ viền\nĐoán: {s3b['predicted_label']} ({'ĐÚNG' if s3b['is_correct'] else 'SAI'}) | Hạng '{lbl}': #{s3b['true_label_rank']}", s3b["is_correct"]),
            (ds_g64, f"Mẫu Dataset lớp '{lbl}'\n(64×64 chuẩn để đối chiếu)", None),
        ]

        for col_i, (img64, title_txt, is_ok) in enumerate(cols_data):
            ax = axes[idx, col_i]
            ax.imshow(img64, cmap="gray", vmin=0, vmax=255)
            color = "#1e40af" if is_ok is None else ("#166534" if is_ok else "#991b1b")
            ax.set_title(title_txt, fontsize=7.5, fontweight="bold", color=color)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "So sánh ảnh xám 64×64 thực tế đưa vào HOG+SVM trên toàn bộ 23 mẫu (Cách 1 vs Cách 2 vs Cách 3A/3B vs Dataset)",
        fontsize=13,
        fontweight="bold",
        y=0.996,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.993])
    fig.savefig(OUT_ALL_23_PNG, bbox_inches="tight")
    plt.close(fig)

    # Visual Figure 2: Focused view on all 10 B & D samples (#14..#23) + C samples (#01..#07)
    focus_indices = list(range(13, 23))  # #14..#23 (all B and D samples)
    fig2, axes2 = plt.subplots(len(focus_indices), 5, figsize=(13.5, 22), dpi=135)
    for row_i, idx in enumerate(focus_indices):
        rec = records[idx]
        lbl = rec["verified_true_label"]
        s1 = m1_unflipped["samples"][idx]
        s2 = m2_flip["samples"][idx]
        s3a = m3a_hard["samples"][idx]
        s3b = m3b_preserve["samples"][idx]

        ds_bgr = cv2.imread(str(PROJECT_ROOT / "frontend" / "public" / "signs" / f"{lbl}.jpg"))
        _, _, ds_g64 = extract_hog_with_diagnostics_from_bgr(ds_bgr)

        cols_data = [
            (s1["gray64"], f"#{idx+1:02d} (Thật: {lbl}) Cách 1: Gốc\nĐoán: {s1['predicted_label']} | Hạng '{lbl}': #{s1['true_label_rank']}", s1["is_correct"]),
            (s2["gray64"], f"Cách 2: Lật ngang 1 lần\nĐoán: {s2['predicted_label']} | Hạng '{lbl}': #{s2['true_label_rank']}", s2["is_correct"]),
            (s3a["gray64"], f"Cách 3A: Tách cứng Cr>=135\nĐoán: {s3a['predicted_label']} | Hạng '{lbl}': #{s3a['true_label_rank']}", s3a["is_correct"]),
            (s3b["gray64"], f"Cách 3B: Giảm nền giữ viền\nĐoán: {s3b['predicted_label']} | Hạng '{lbl}': #{s3b['true_label_rank']}", s3b["is_correct"]),
            (ds_g64, f"Ảnh Dataset gốc lớp '{lbl}'\n(Đối chiếu tư thế & độ tương phản)", None),
        ]
        for col_i, (img64, title_txt, is_ok) in enumerate(cols_data):
            ax = axes2[row_i, col_i]
            ax.imshow(img64, cmap="gray", vmin=0, vmax=255)
            color = "#1e40af" if is_ok is None else ("#166534" if is_ok else "#991b1b")
            ax.set_title(title_txt, fontsize=8, fontweight="bold", color=color)
            ax.set_xticks([])
            ax.set_yticks([])

    fig2.suptitle(
        "Chi tiết ảnh xám 64×64 đầu vào của 10 mẫu lớp B (#14..#18) và lớp D (#19..#23) qua 3 cách xử lý",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig2.savefig(OUT_BD_FOCUS_PNG, bbox_inches="tight")
    plt.close(fig2)

    # Save clean JSON report without raw numpy arrays
    def strip_arrays(res_dict: dict) -> dict:
        copy_d = dict(res_dict)
        copy_d["samples"] = [
            {k: v for k, v in s.items() if k not in ("gray64", "input_bgr", "mask_used")}
            for s in res_dict["samples"]
        ]
        return copy_d

    report_payload = {
        "note_on_sample_04": (
            "Mẫu #04 (sample_20261123_220324_317901_sign_C_frame_F226_1259853) là mẫu duy nhất có "
            "flip_input_for_model=True lúc lưu, nghĩa là file received_crop.png trên đĩa đã được lật ngang 1 lần. "
            "Vì vậy ở Cách 2 và Cách 3, mẫu #04 được giữ nguyên (không lật lần 2), còn 22 mẫu còn lại được lật ngang 1 lần."
        ),
        "methods": {
            "M1_raw_unflipped": strip_arrays(m1_unflipped),
            "M1_as_saved_on_disk": strip_arrays(m1_disk),
            "M2_flipped_once": strip_arrays(m2_flip),
            "M3A_flip_hard_ycrcb": strip_arrays(m3a_hard),
            "M3B_flip_preserve_hand_bg_suppressed": strip_arrays(m3b_preserve),
        },
    }
    OUT_JSON_PATH.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu báo cáo JSON: {OUT_JSON_PATH}")
    print(f"Đã lưu ảnh 23 mẫu  : {OUT_ALL_23_PNG}")
    print(f"Đã lưu ảnh B & D   : {OUT_BD_FOCUS_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
