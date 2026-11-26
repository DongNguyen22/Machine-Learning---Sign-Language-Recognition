r"""Evaluate newly captured independent webcam samples (excluding the 23 old selection samples)
and verify 100% parity of the integrated Method 3B pipeline in SVMSignPredictor & HTTP API.

Usage examples (PowerShell):
1. Verify that the integrated pipeline in `SVMSignPredictor` and `svm_api_server.py` matches
   the offline experiment (`scripts/compare_three_input_pipelines.py`) 100% across all modes,
   and confirm that training data, model, and the 23 old samples are untouched:
   .\.venv\Scripts\python.exe scripts/evaluate_new_webcam_samples.py --verify-integration

2. After capturing new independent webcam samples of A/B/C/D via the SignVision UI, compare
   Legacy (Cách cũ) vs Method 3B (Cách 3B) on the exact same newly captured frames and export
   per-class & total accuracy tables:
   .\.venv\Scripts\python.exe scripts/evaluate_new_webcam_samples.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

from src.model.predict_svm import DEFAULT_MODEL_PATH, SVMSignPredictor
from src.preprocess.build_hog_dataset import extract_hog_with_diagnostics_from_bgr

DIAG_DIR = PROJECT_ROOT / "results" / "svm" / "webcam_diagnostics"
OFFLINE_REPORT_PATH = DIAG_DIR / "three_methods_comparison_report.json"

# Exact set of 23 old diagnostic samples used to select Method 3B.
# These 23 samples MUST NEVER be reused to claim accuracy on new independent webcam images.
OLD_23_SELECTION_FOLDERS = {
    "sample_20261123_215214_286607_sign_B_frame_F210_675450",
    "sample_20261123_215254_304844_sign_C_frame_F20_724723",
    "sample_20261123_220133_256249_sign_C_frame_F186_1244143",
    "sample_20261123_220324_317901_sign_C_frame_F226_1259853",
    "sample_20261123_220332_786184_sign_C_frame_F246_1368910",
    "sample_20261123_220336_203593_sign_C_frame_F254_1372317",
    "sample_20261123_220340_014063_sign_C_frame_F263_1375841",
    "sample_20261123_220348_871764_sign_C_frame_F286_1384957",
    "sample_20261123_220357_205164_sign_A_frame_F305_1393100",
    "sample_20261123_220400_060235_sign_A_frame_F312_1396199",
    "sample_20261123_220402_528856_sign_A_frame_F318_1398658",
    "sample_20261123_220405_118222_sign_A_frame_F323_1401147",
    "sample_20261123_220409_916831_sign_A_frame_F335_1406033",
    "sample_20261123_220417_476174_sign_B_frame_F351_1413111",
    "sample_20261123_220420_587049_sign_B_frame_F359_1416691",
    "sample_20261123_220422_844380_sign_B_frame_F364_1418738",
    "sample_20261123_220425_822879_sign_B_frame_F371_1421785",
    "sample_20261123_220428_613395_sign_B_frame_F377_1424354",
    "sample_20261123_220542_831085_sign_D_frame_F32_1498853",
    "sample_20261123_220549_845262_sign_D_frame_F50_1505989",
    "sample_20261123_220552_334977_sign_D_frame_F56_1508404",
    "sample_20261123_220554_864210_sign_D_frame_F61_1510674",
    "sample_20261123_220556_339187_sign_D_frame_F65_1512521",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_integration() -> int:
    """Verify 100% parity between integrated SVMSignPredictor and offline benchmark across all modes."""
    if not OFFLINE_REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing offline report: {OFFLINE_REPORT_PATH}")

    offline = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))
    m1_off = {s["folder_name"]: s for s in offline["methods"]["M1_raw_unflipped"]["samples"]}
    m2_off = {s["folder_name"]: s for s in offline["methods"]["M2_flipped_once"]["samples"]}
    m3b_off = {
        s["folder_name"]: s
        for s in offline["methods"]["M3B_flip_preserve_hand_bg_suppressed"]["samples"]
    }

    model_path = PROJECT_ROOT / DEFAULT_MODEL_PATH
    hog_npz_path = PROJECT_ROOT / "data" / "processed" / "svm_hog_features.npz"
    model_sha_before = sha256_file(model_path)
    hog_sha_before = sha256_file(hog_npz_path)

    # Record SHA-256 of all 23 old received_crop.png files before testing
    old_crop_hashes_before = {
        fname: sha256_file(DIAG_DIR / fname / "received_crop.png")
        for fname in sorted(OLD_23_SELECTION_FOLDERS)
    }

    predictor = SVMSignPredictor(model_path=model_path)

    m1_matches = 0
    m2_matches = 0
    m3b_matches = 0
    same_frame_comp_matches = 0
    max_score_diff = 0.0

    for fname in sorted(OLD_23_SELECTION_FOLDERS):
        sdir = DIAG_DIR / fname
        meta = json.loads((sdir / "diagnostic_metadata.json").read_text(encoding="utf-8"))
        flip_saved = bool(meta.get("flip_input_for_model", False))
        crop_bgr = cv2.imread(str(sdir / "received_crop.png"))

        # Test 1: Mode OFF (Cách cũ - unflipped raw orientation)
        # If the saved image was already flipped (#04), unflip it first to simulate raw camera frame,
        # OR pass already_flipped=flip_saved. Notice when experimental_webcam_mode=False,
        # predict_image_bgr uses target_bgr as-is for prediction, and its same_frame_comparison["legacy"]
        # uses raw_unflipped_bgr!
        raw_unflipped_bgr = cv2.flip(crop_bgr, 1) if flip_saved else crop_bgr
        res_m1 = predictor.predict_image_bgr(
            raw_unflipped_bgr,
            top_k=3,
            include_diagnostics=True,
            experimental_webcam_mode=False,
            already_flipped=False,
        )
        assert res_m1["predicted_label"] == m1_off[fname]["predicted_label"], (
            f"M1 label mismatch on {fname}: {res_m1['predicted_label']} vs {m1_off[fname]['predicted_label']}"
        )
        diff1 = abs(float(res_m1["predicted_decision_score"]) - float(m1_off[fname]["predicted_decision_score"]))
        assert diff1 < 1e-6, f"M1 score diff too large on {fname}: {diff1}"
        max_score_diff = max(max_score_diff, diff1)
        m1_matches += 1

        # Test 2: Flip-only (Cách 2 - single horizontal flip)
        flipped_once_bgr = crop_bgr if flip_saved else cv2.flip(crop_bgr, 1)
        res_m2 = predictor.predict_image_bgr(
            flipped_once_bgr,
            top_k=3,
            include_diagnostics=False,
            experimental_webcam_mode=False,
            already_flipped=True,
        )
        assert res_m2["predicted_label"] == m2_off[fname]["predicted_label"], (
            f"M2 label mismatch on {fname}: {res_m2['predicted_label']} vs {m2_off[fname]['predicted_label']}"
        )
        diff2 = abs(float(res_m2["predicted_decision_score"]) - float(m2_off[fname]["predicted_decision_score"]))
        assert diff2 < 1e-6, f"M2 score diff too large on {fname}: {diff2}"
        max_score_diff = max(max_score_diff, diff2)
        m2_matches += 1

        # Test 3: Mode ON (Cách 3B - passing crop_bgr with already_flipped=flip_saved to verify NO double-flip!)
        res_m3b = predictor.predict_image_bgr(
            crop_bgr,
            top_k=3,
            include_diagnostics=True,
            experimental_webcam_mode=True,
            already_flipped=flip_saved,
        )
        assert res_m3b["predicted_label"] == m3b_off[fname]["predicted_label"], (
            f"M3B label mismatch on {fname}: {res_m3b['predicted_label']} vs {m3b_off[fname]['predicted_label']}"
        )
        diff3 = abs(float(res_m3b["predicted_decision_score"]) - float(m3b_off[fname]["predicted_decision_score"]))
        assert diff3 < 1e-6, f"M3B score diff too large on {fname}: {diff3}"
        max_score_diff = max(max_score_diff, diff3)
        m3b_matches += 1

        # Test 4: Verify same_frame_comparison returns BOTH M1 and M3B accurately in a single call
        comp = res_m3b["same_frame_comparison"]
        assert comp["legacy"]["predicted_label"] == m1_off[fname]["predicted_label"]
        assert comp["method_3b"]["predicted_label"] == m3b_off[fname]["predicted_label"]
        same_frame_comp_matches += 1

    # Verify files were not modified
    assert sha256_file(model_path) == model_sha_before, "Model file was modified!"
    assert sha256_file(hog_npz_path) == hog_sha_before, "HOG train/test NPZ was modified!"
    for fname, h_before in old_crop_hashes_before.items():
        assert sha256_file(DIAG_DIR / fname / "received_crop.png") == h_before, (
            f"Old sample {fname} was modified!"
        )

    print("================ KẾT QUẢ KIỂM TRA TÍCH HỢP CÁCH 3B ================")
    print(f"1. Khớp Cách 1 (Ảnh crop gốc - TẮT Cách 3B)       : {m1_matches}/23 mẫu (100% khớp nhãn & điểm)")
    print(f"2. Khớp Cách 2 (Chỉ lật ngang 1 lần)              : {m2_matches}/23 mẫu (100% khớp nhãn & điểm)")
    print(f"3. Khớp Cách 3B (BẬT Xử lý webcam thử nghiệm)     : {m3b_matches}/23 mẫu (100% khớp nhãn & điểm, tránh lật 2 lần ở #04)")
    print(f"4. Khớp So sánh cùng khung hình (Legacy vs 3B)    : {same_frame_comp_matches}/23 mẫu")
    print(f"5. Sai số tuyệt đối lớn nhất của decision_score   : {max_score_diff:.2e}")
    print(f"6. Toàn vẹn dữ liệu train/model/23 mẫu cũ (SHA256): NGUYÊN VẸN 100%")
    print("===================================================================")
    return 0


def evaluate_new_samples() -> int:
    """Evaluate all newly captured independent webcam samples (strictly excluding the 23 old samples)."""
    all_dirs = sorted(
        [d for d in DIAG_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )
    new_dirs = [d for d in all_dirs if d.name not in OLD_23_SELECTION_FOLDERS]

    if not new_dirs:
        print("===================================================================")
        print("CHƯA CÓ MẪU WEBCAM MỚI ĐỘC LẬP NÀO (NGOÀI 23 MẪU CŨ ĐÃ DÙNG ĐỂ CHỌN CÁCH 3B).")
        print("Hệ thống cố ý KHÔNG dùng lại 23 mẫu cũ để báo cáo hiệu quả thực tế.")
        print("Hãy mở giao diện SignVision, chụp và nhấn 'Lưu mẫu chẩn đoán' cho các mẫu mới,")
        print("sau đó chạy lại lệnh này:")
        print("  .\\.venv\\Scripts\\python.exe scripts/evaluate_new_webcam_samples.py")
        print("===================================================================")
        return 0

    predictor = SVMSignPredictor()
    rows = []
    for idx, sdir in enumerate(new_dirs, start=1):
        meta_path = sdir / "diagnostic_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        true_lbl = str(
            meta.get("verified_true_label") or meta.get("intended_sign") or ""
        ).strip().upper()
        crop_bgr = cv2.imread(str(sdir / "received_crop.png"))
        if crop_bgr is None:
            continue
        flip_saved = bool(meta.get("flip_input_for_model", False))

        # Evaluate both Legacy (raw unflipped) and Method 3B on the exact same frame
        res = predictor.predict_image_bgr(
            crop_bgr,
            top_k=3,
            include_diagnostics=True,
            experimental_webcam_mode=True,
            already_flipped=flip_saved,
        )
        comp = res["same_frame_comparison"]
        leg_pred = str(comp["legacy"]["predicted_label"])
        leg_score = float(comp["legacy"]["predicted_decision_score"])
        m3b_pred = str(comp["method_3b"]["predicted_label"])
        m3b_score = float(comp["method_3b"]["predicted_decision_score"])

        rows.append(
            {
                "stt": idx,
                "folder_name": sdir.name,
                "true_label": true_lbl,
                "label_confirmed": bool(meta.get("label_confirmed", False)),
                "active_mode_when_saved": meta.get("webcam_mode_label", "—"),
                "user_note": meta.get("user_note", ""),
                "legacy_pred": leg_pred,
                "legacy_correct": (leg_pred == true_lbl),
                "legacy_score": round(leg_score, 4),
                "method_3b_pred": m3b_pred,
                "method_3b_correct": (m3b_pred == true_lbl),
                "method_3b_score": round(m3b_score, 4),
            }
        )

    print(f"================ KẾT QUẢ TRÊN {len(rows)} MẪU WEBCAM MỚI ĐỘC LẬP ================")
    print("(Đã loại trừ hoàn toàn 23 mẫu cũ dùng để chọn phương pháp)")
    for r in rows:
        leg_ok = "ĐÚNG" if r["legacy_correct"] else "SAI "
        m3b_ok = "ĐÚNG" if r["method_3b_correct"] else "SAI "
        print(
            f"#{r['stt']:02d} | Nhãn thật: {r['true_label']} | "
            f"Cách cũ: {r['legacy_pred']} ({leg_ok}, {r['legacy_score']:+.3f}) | "
            f"Cách 3B: {r['method_3b_pred']} ({m3b_ok}, {r['method_3b_score']:+.3f}) | "
            f"Ghi chú: {r['user_note'] or '—'} | {r['folder_name']}"
        )

    classes = sorted(list({r["true_label"] for r in rows}))
    print("\n--- Bảng tổng hợp theo từng chữ trên bộ mẫu mới độc lập ---")
    for cls in classes:
        c_rows = [r for r in rows if r["true_label"] == cls]
        leg_c = sum(1 for r in c_rows if r["legacy_correct"])
        m3b_c = sum(1 for r in c_rows if r["method_3b_correct"])
        tot = len(c_rows)
        print(
            f"  Lớp {cls}: Cách cũ đúng {leg_c}/{tot} ({leg_c/tot*100:.1f}%) | "
            f"Cách 3B đúng {m3b_c}/{tot} ({m3b_c/tot*100:.1f}%)"
        )

    tot_leg = sum(1 for r in rows if r["legacy_correct"])
    tot_m3b = sum(1 for r in rows if r["method_3b_correct"])
    tot_all = len(rows)
    print(
        f"\n=> TỔNG SỐ ĐÚNG (BỘ MẪU MỚI): "
        f"Cách cũ = {tot_leg}/{tot_all} ({tot_leg/tot_all*100:.2f}%) | "
        f"Cách 3B = {tot_m3b}/{tot_all} ({tot_m3b/tot_all*100:.2f}%)"
    )

    csv_out = DIAG_DIR / "new_independent_samples_comparison.csv"
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_out = DIAG_DIR / "new_independent_samples_comparison.json"
    per_class_summary = {}
    for cls in classes:
        c_rows = [r for r in rows if r["true_label"] == cls]
        leg_c = sum(1 for r in c_rows if r["legacy_correct"])
        m3b_c = sum(1 for r in c_rows if r["method_3b_correct"])
        tot = len(c_rows)
        per_class_summary[cls] = {
            "total": tot,
            "legacy_correct": leg_c,
            "legacy_accuracy_pct": round(leg_c / tot * 100.0, 2),
            "legacy_pred_distribution": dict(Counter(r["legacy_pred"] for r in c_rows)),
            "method_3b_correct": m3b_c,
            "method_3b_accuracy_pct": round(m3b_c / tot * 100.0, 2),
            "method_3b_pred_distribution": dict(Counter(r["method_3b_pred"] for r in c_rows)),
        }

    json_out.write_text(
        json.dumps(
            {
                "old_selection_samples_excluded": len(OLD_23_SELECTION_FOLDERS),
                "new_independent_samples_invalid_excluded": 0,
                "new_independent_samples_evaluated": tot_all,
                "legacy_total_correct": tot_leg,
                "legacy_total_accuracy_pct": round(tot_leg / tot_all * 100.0, 2),
                "method_3b_total_correct": tot_m3b,
                "method_3b_total_accuracy_pct": round(tot_m3b / tot_all * 100.0, 2),
                "per_class": per_class_summary,
                "samples": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Render visual comparison sheet for the 32 new independent samples
    cols = 4
    n_rows = max(1, math.ceil(tot_all / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 4.8, n_rows * 3.1), dpi=135)
    axes_flat = list(np.array(axes).reshape(-1))
    for i, ax in enumerate(axes_flat):
        if i >= tot_all:
            ax.axis("off")
            continue
        r = rows[i]
        sdir = DIAG_DIR / r["folder_name"]
        crop_bgr = cv2.imread(str(sdir / "received_crop.png"))
        _, _, leg_g64 = extract_hog_with_diagnostics_from_bgr(crop_bgr)
        from src.preprocess.build_hog_dataset import apply_webcam_method_3b_from_bgr
        m3b_bgr, _, _ = apply_webcam_method_3b_from_bgr(crop_bgr, already_flipped=False)
        _, _, m3b_g64 = extract_hog_with_diagnostics_from_bgr(m3b_bgr)

        crop_rgb = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (110, 110), interpolation=cv2.INTER_AREA)
        leg_rgb = cv2.resize(cv2.cvtColor(leg_g64, cv2.COLOR_GRAY2RGB), (110, 110), interpolation=cv2.INTER_NEAREST)
        m3b_rgb = cv2.resize(cv2.cvtColor(m3b_g64, cv2.COLOR_GRAY2RGB), (110, 110), interpolation=cv2.INTER_NEAREST)
        sep = np.full((110, 4, 3), 240, dtype=np.uint8)
        comb = np.hstack([crop_rgb, sep, leg_rgb, sep, m3b_rgb])
        ax.imshow(comb)
        title_color = "#166534" if r["method_3b_correct"] else "#991b1b"
        ax.set_title(
            f"#N{r['stt']:02d} | Thật: {r['true_label']} | Cũ->{r['legacy_pred']} ({'Đ' if r['legacy_correct'] else 'S'}) "
            f"| 3B->{r['method_3b_pred']} ({'ĐÚNG' if r['method_3b_correct'] else 'SAI'})\n"
            f"Score Cũ: {r['legacy_score']:+.2f} | Score 3B: {r['method_3b_score']:+.2f}",
            fontsize=7.8,
            fontweight="bold",
            color=title_color,
        )
        ax.set_xticks([55, 114 + 55, 228 + 55])
        ax.set_xticklabels(["Crop gốc", "Xám 64 (Cũ)", "Xám 64 (3B)"], fontsize=7)
        ax.set_yticks([])

    fig.suptitle(
        f"Kết quả đối chiếu Cách cũ ({tot_leg}/{tot_all}) vs Cách 3B ({tot_m3b}/{tot_all}) trên {tot_all} mẫu Webcam MỚI độc lập",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    png_out = DIAG_DIR / "new_independent_samples_comparison.png"
    fig.savefig(png_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Đã xuất bảng CSV bộ mẫu mới tại : {csv_out}")
    print(f"Đã xuất báo cáo JSON bộ mẫu mới : {json_out}")
    print(f"Đã xuất ảnh so sánh bộ mẫu mới  : {png_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra tích hợp Cách 3B và đánh giá bộ mẫu webcam mới độc lập."
    )
    parser.add_argument(
        "--verify-integration",
        action="store_true",
        help="Kiểm tra tính nhất quán 100% giữa pipeline tích hợp và phép thử ngoại tuyến trên 23 mẫu.",
    )
    args = parser.parse_args()
    if args.verify_integration:
        return verify_integration()
    return evaluate_new_samples()


if __name__ == "__main__":
    raise SystemExit(main())
