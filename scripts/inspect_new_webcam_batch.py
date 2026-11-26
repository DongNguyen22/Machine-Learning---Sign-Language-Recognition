r"""Inspect and generate a visual review contact sheet & metadata audit for newly captured
webcam samples (strictly excluding the 23 old selection samples), WITHOUT computing accuracy
or auto-confirming labels before user verification.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.build_hog_dataset import (
    apply_webcam_method_3b_from_bgr,
    extract_hog_with_diagnostics_from_bgr,
)
from scripts.evaluate_new_webcam_samples import OLD_23_SELECTION_FOLDERS

DIAG_DIR = PROJECT_ROOT / "results" / "svm" / "webcam_diagnostics"
NEW_CONTACT_SHEET_PNG = DIAG_DIR / "new_batch_review_contact_sheet.png"
NEW_REVIEW_CSV = DIAG_DIR / "new_batch_label_review_table.csv"
NEW_REVIEW_JSON = DIAG_DIR / "new_batch_audit_summary.json"


def main() -> int:
    all_dirs = sorted(
        [d for d in DIAG_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )
    old_dirs = [d for d in all_dirs if d.name in OLD_23_SELECTION_FOLDERS]
    new_dirs = [d for d in all_dirs if d.name not in OLD_23_SELECTION_FOLDERS]

    print(f"Tổng số thư mục mẫu trong {DIAG_DIR.relative_to(PROJECT_ROOT)}: {len(all_dirs)}")
    print(f"- Số mẫu cũ (đợt chọn Cách 3B, đã tách riêng): {len(old_dirs)}")
    print(f"- Số mẫu MỚI (đợt chụp kiểm chứng độc lập)   : {len(new_dirs)}")

    if not new_dirs:
        print("Không tìm thấy mẫu mới nào!")
        return 1

    records: list[dict] = []
    for idx, sdir in enumerate(new_dirs, start=1):
        meta_path = sdir / "diagnostic_metadata.json"
        crop_path = sdir / "received_crop.png"
        gray_path = sdir / "preprocessed_gray_64x64.png"

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        crop_bgr = cv2.imread(str(crop_path))
        saved_gray64 = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)

        intended_sign = str(meta.get("intended_sign", "")).strip().upper()
        flip_saved = bool(meta.get("flip_input_for_model", False))
        preview_mirrored = bool(meta.get("preview_mirrored", True))
        user_note = str(meta.get("user_note", "")).strip()
        saved_at = str(meta.get("saved_at", ""))
        frame_id = str(meta.get("frame_id", ""))
        roi_mode = str(meta.get("roi_mode", ""))
        crop_box = meta.get("crop_box", {})

        # Check explicit metadata fields for Method 3B
        has_new_meta_fields = ("experimental_webcam_mode" in meta) or ("webcam_mode" in meta)
        meta_exp_3b = meta.get("experimental_webcam_mode", None)
        meta_webcam_mode = meta.get("webcam_mode", None)
        meta_webcam_mode_label = meta.get("webcam_mode_label", None)
        has_same_frame_comp = isinstance(meta.get("same_frame_comparison"), dict)

        # Mathematically verify whether saved_gray64 matches:
        # (A) Legacy gray64 from received_crop.png as-is
        # (B) Method 3B gray64 from received_crop.png (with already_flipped=flip_saved)
        _, _, calc_legacy_gray64 = extract_hog_with_diagnostics_from_bgr(crop_bgr)
        m3b_bgr, _, _ = apply_webcam_method_3b_from_bgr(crop_bgr, already_flipped=flip_saved)
        _, _, calc_m3b_gray64 = extract_hog_with_diagnostics_from_bgr(m3b_bgr)

        diff_to_legacy = float(np.max(np.abs(saved_gray64.astype(np.int32) - calc_legacy_gray64.astype(np.int32))))
        diff_to_m3b = float(np.max(np.abs(saved_gray64.astype(np.int32) - calc_m3b_gray64.astype(np.int32))))

        if diff_to_m3b == 0.0 and diff_to_legacy > 0.0:
            verified_saved_gray_source = "method_3b"
        elif diff_to_legacy == 0.0 and diff_to_m3b > 0.0:
            verified_saved_gray_source = "legacy"
        elif diff_to_m3b < diff_to_legacy:
            verified_saved_gray_source = "method_3b_approx"
        else:
            verified_saved_gray_source = "legacy_approx"

        h, w = crop_bgr.shape[:2]
        # Check if crop touches frame boundaries (640x480 typical webcam resolution)
        bx = int(crop_box.get("x", 0)) if isinstance(crop_box, dict) else 0
        by = int(crop_box.get("y", 0)) if isinstance(crop_box, dict) else 0
        bw = int(crop_box.get("width", w)) if isinstance(crop_box, dict) else w
        bh = int(crop_box.get("height", h)) if isinstance(crop_box, dict) else h
        touches_border = []
        if bx <= 1:
            touches_border.append("trái")
        if by <= 1:
            touches_border.append("trên")
        if bx + bw >= 639:
            touches_border.append("phải")
        if by + bh >= 479:
            touches_border.append("dưới")

        records.append(
            {
                "new_stt": idx,
                "folder_name": sdir.name,
                "saved_at": saved_at,
                "frame_id": frame_id,
                "intended_sign": intended_sign,
                "label_confirmed": bool(meta.get("label_confirmed", False)),
                "verified_true_label": meta.get("verified_true_label", None),
                "has_new_meta_fields": has_new_meta_fields,
                "meta_experimental_webcam_mode": meta_exp_3b,
                "meta_webcam_mode": meta_webcam_mode,
                "meta_webcam_mode_label": meta_webcam_mode_label,
                "has_same_frame_comparison_in_meta": has_same_frame_comp,
                "verified_saved_gray64_pipeline": verified_saved_gray_source,
                "max_pixel_diff_to_legacy": diff_to_legacy,
                "max_pixel_diff_to_m3b": diff_to_m3b,
                "flip_input_for_model": flip_saved,
                "preview_mirrored": preview_mirrored,
                "user_note": user_note,
                "roi_mode": roi_mode,
                "crop_box": crop_box,
                "crop_width": w,
                "crop_height": h,
                "touches_frame_border": touches_border,
                "can_recompute_both_modes_fairly": True,
                "crop_rel_path": crop_path.relative_to(PROJECT_ROOT).as_posix(),
                "gray_rel_path": gray_path.relative_to(PROJECT_ROOT).as_posix(),
                "crop_bgr": crop_bgr,
                "saved_gray64": saved_gray64,
                "calc_legacy_gray64": calc_legacy_gray64,
                "calc_m3b_gray64": calc_m3b_gray64,
            }
        )

    # Generate contact sheet showing for each new sample:
    # 1. Ảnh crop gốc (received_crop.png)
    # 2. Ảnh xám 64x64 đã lưu lúc chụp (preprocessed_gray_64x64.png)
    # 3. Ảnh xám 64x64 Cách 3B (để kiểm tra viền tay/ngón tay)
    n_new = len(records)
    cols = 3
    rows = max(1, math.ceil(n_new / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.6, rows * 3.4), dpi=140)
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).reshape(-1))

    for idx, ax in enumerate(axes_flat):
        if idx >= n_new:
            ax.axis("off")
            continue
        rec = records[idx]
        crop_rgb = cv2.cvtColor(rec["crop_bgr"], cv2.COLOR_BGR2RGB)
        crop_disp = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_AREA)

        saved_g_rgb = cv2.cvtColor(rec["saved_gray64"], cv2.COLOR_GRAY2RGB)
        saved_g_disp = cv2.resize(saved_g_rgb, (128, 128), interpolation=cv2.INTER_NEAREST)

        m3b_g_rgb = cv2.cvtColor(rec["calc_m3b_gray64"], cv2.COLOR_GRAY2RGB)
        m3b_g_disp = cv2.resize(m3b_g_rgb, (128, 128), interpolation=cv2.INTER_NEAREST)

        sep = np.full((128, 5, 3), 240, dtype=np.uint8)
        combined = np.hstack([crop_disp, sep, saved_g_disp, sep, m3b_g_disp])

        mode_saved_txt = (
            "3B: BẬT"
            if rec["verified_saved_gray64_pipeline"].startswith("method_3b")
            else "3B: TẮT (Lưu ảnh xám gốc)"
        )
        flip_txt = "Lật=BẬT" if rec["flip_input_for_model"] else "Lật=TẮT"
        note_txt = f" | Ghi chú: {rec['user_note']}" if rec["user_note"] else ""

        ax.imshow(combined)
        ax.set_title(
            f"#N{rec['new_stt']:02d} — Chữ đã chọn: '{rec['intended_sign']}' | {mode_saved_txt} | {flip_txt}{note_txt}\n"
            f"{rec['folder_name']} (crop {rec['crop_width']}×{rec['crop_height']})",
            fontsize=8,
            fontweight="bold",
        )
        ax.set_xticks([64, 133 + 64, 266 + 64])
        ax.set_xticklabels(
            ["1. Ảnh crop gốc", "2. Xám 64×64 lúc lưu", "3. Xám 64×64 Cách 3B"],
            fontsize=7.5,
        )
        ax.set_yticks([])

    fig.suptitle(
        f"Bảng rà soát {n_new} mẫu Webcam MỚI (Đánh số #N01..#N{n_new:02d} — Tách riêng khỏi 23 mẫu cũ, CHƯA tính độ chính xác)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.968])
    fig.savefig(NEW_CONTACT_SHEET_PNG, bbox_inches="tight")
    plt.close(fig)

    # Save CSV & JSON summary
    with NEW_REVIEW_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "new_stt",
            "folder_name",
            "intended_sign",
            "verified_saved_gray64_pipeline",
            "meta_experimental_webcam_mode",
            "flip_input_for_model",
            "user_note",
            "crop_width",
            "crop_height",
            "touches_frame_border",
            "saved_at",
            "crop_rel_path",
            "gray_rel_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r[k] for k in fieldnames}
            row["touches_frame_border"] = ",".join(r["touches_frame_border"])
            writer.writerow(row)

    clean_records = [
        {k: v for k, v in r.items() if k not in ("crop_bgr", "saved_gray64", "calc_legacy_gray64", "calc_m3b_gray64")}
        for r in records
    ]
    NEW_REVIEW_JSON.write_text(
        json.dumps(
            {
                "old_samples_excluded_count": len(old_dirs),
                "new_samples_count": len(clean_records),
                "new_samples": clean_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n================ DANH SÁCH MẪU MỚI ĐỢT KIỂM CHỨNG ================")
    for r in clean_records:
        print(
            f"#N{r['new_stt']:02d} | {r['folder_name']} | Chữ chọn: '{r['intended_sign']}' | "
            f"Ảnh xám lúc lưu: {r['verified_saved_gray64_pipeline']} (meta 3B={r['meta_experimental_webcam_mode']}) | "
            f"flip_saved={r['flip_input_for_model']} | crop={r['crop_width']}x{r['crop_height']} "
            f"(chạm biên: {r['touches_frame_border'] or 'không'}) | Ghi chú: '{r['user_note']}'"
        )
    print("===================================================================")
    print(f"Đã tạo ảnh tổng hợp mẫu mới: {NEW_CONTACT_SHEET_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
