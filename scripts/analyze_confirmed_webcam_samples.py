r"""Analyze the 23 confirmed webcam diagnostic samples (A, B, C, D) without retraining.

Rules enforced:
1. Only analyzes samples with `label_confirmed == True`.
2. Ground truth is strictly `verified_true_label` from `diagnostic_metadata.json`
   (never `folder_name`, `original_intended_sign`, or model prediction).
3. Computes per-class accuracy, total accuracy (/23), and confusion table
   (verified_true_label -> raw_predicted_label).
4. Checks saved `flip_input_for_model` status across samples and evaluates whether
   matched pairs exist, plus measures counterfactual horizontal flip on the saved crops
   purely as diagnostic evidence without overclaiming.
5. Compares webcam 64x64 preprocessed images & HOG features against training dataset
   reference samples for A, B, C, D to separate crop, background/lighting, hand orientation,
   and model generalization factors.
6. Does NOT train, modify, or fine-tune any model.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
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
REPORT_JSON_PATH = DIAG_DIR / "confirmed_23_analysis_report.json"
REPORT_MD_PATH = DIAG_DIR / "confirmed_23_analysis_report.md"
MISCLASSIFIED_PNG_PATH = DIAG_DIR / "misclassified_samples_contact_sheet.png"
COMPARISON_PNG_PATH = DIAG_DIR / "webcam_vs_dataset_diagnostic_comparison.png"


def load_confirmed_samples() -> list[dict]:
    if not DIAG_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {DIAG_DIR}")

    sample_dirs = sorted(
        [d for d in DIAG_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")],
        key=lambda d: d.name,
    )

    records: list[dict] = []
    for idx, sdir in enumerate(sample_dirs, start=1):
        meta_path = sdir / "diagnostic_metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not bool(meta.get("label_confirmed", False)):
            raise ValueError(f"Sample #{idx:02d} ({sdir.name}) is not confirmed!")
        true_label = str(meta["verified_true_label"]).strip().upper()
        raw_pred = str(meta.get("raw_predicted_label", "")).strip().upper()
        smoothed_pred = str(meta.get("smoothed_label", "")).strip().upper()
        crop_path = sdir / "received_crop.png"
        gray_path = sdir / "preprocessed_gray_64x64.png"

        records.append(
            {
                "stt": idx,
                "folder_name": sdir.name,
                "verified_true_label": true_label,
                "original_intended_sign": str(meta.get("original_intended_sign", "")).strip().upper(),
                "raw_predicted_label": raw_pred,
                "smoothed_label": smoothed_pred,
                "is_correct_raw": (true_label == raw_pred),
                "predicted_decision_score": float(meta.get("predicted_decision_score", 0.0)),
                "margin_top1_minus_top2": float(meta.get("margin_top1_minus_top2", 0.0)),
                "top_k": meta.get("top_k", []),
                "roi_mode": meta.get("roi_mode", ""),
                "crop_box": meta.get("crop_box", {}),
                "flip_input_for_model": bool(meta.get("flip_input_for_model", False)),
                "preview_mirrored": bool(meta.get("preview_mirrored", True)),
                "image_dimensions": meta.get("image_dimensions", {}),
                "gray_64x64_stats": meta.get("gray_64x64_stats", {}),
                "hog_l2_norm": float(meta.get("hog_l2_norm", 0.0)),
                "saved_at": meta.get("saved_at", ""),
                "frame_id": meta.get("frame_id", ""),
                "crop_path": crop_path,
                "gray_path": gray_path,
                "crop_rel_path": crop_path.relative_to(PROJECT_ROOT).as_posix(),
                "gray_rel_path": gray_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    return records


def analyze_with_predictor(records: list[dict], predictor: SVMSignPredictor) -> dict:
    # Load dataset HOG features to compute class centroids in HOG space for A, B, C, D
    hog_npz = np.load(PROJECT_ROOT / "data" / "processed" / "svm_hog_features.npz")
    X_train = hog_npz["X_train"].astype(np.float32)
    y_train = hog_npz["y_train"].astype(np.int64)
    class_names = [str(c) for c in hog_npz["class_names"]]
    label_to_idx = {lbl: i for i, lbl in enumerate(class_names)}

    train_centroids: dict[str, np.ndarray] = {}
    for lbl in ["A", "B", "C", "D", "J", "W", "0", "4", "F", "G"]:
        if lbl in label_to_idx:
            c_idx = label_to_idx[lbl]
            train_centroids[lbl] = X_train[y_train == c_idx].mean(axis=0)

    # Also measure border/background pixel statistics on dataset vs webcam 64x64 images
    for rec in records:
        crop_bgr = cv2.imread(str(rec["crop_path"]))
        if crop_bgr is None:
            raise ValueError(f"Could not read crop image: {rec['crop_path']}")

        # Re-verify prediction with current final model on exact saved crop (no flip)
        pred_unflipped = predictor.predict_image_bgr(crop_bgr, top_k=36)
        # Also run counterfactual horizontal flip on the exact same crop to see score changes
        crop_flipped_bgr = cv2.flip(crop_bgr, 1)
        pred_flipped = predictor.predict_image_bgr(crop_flipped_bgr, top_k=36)

        # Find score and rank of the verified_true_label in full 36-class decision scores
        true_lbl = rec["verified_true_label"]
        unflipped_all = pred_unflipped["top_k_predictions"]
        flipped_all = pred_flipped["top_k_predictions"]

        true_rank_unflipped = next(item["rank"] for item in unflipped_all if item["label"] == true_lbl)
        true_score_unflipped = next(item["decision_score"] for item in unflipped_all if item["label"] == true_lbl)

        true_rank_flipped = next(item["rank"] for item in flipped_all if item["label"] == true_lbl)
        true_score_flipped = next(item["decision_score"] for item in flipped_all if item["label"] == true_lbl)

        # Extract 64x64 gray and HOG vector to analyze padding, contrast, and background gradients
        hog_vec, _resized_bgr_64, gray_64 = extract_hog_with_diagnostics_from_bgr(crop_bgr)

        # Analyze white padding pixels (== 255) vs non-padding border pixels
        white_pad_ratio = float(np.mean(gray_64 == 255))
        # Compute gradient magnitude on gray_64 using Sobel
        gx = cv2.Sobel(gray_64.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_64.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.hypot(gx, gy)

        # Outer 8-pixel frame (excluding pure white padding cols/rows) vs center 48x48
        non_pad_mask = gray_64 < 254
        border_mask = np.zeros_like(non_pad_mask, dtype=bool)
        border_mask[:8, :] = True
        border_mask[-8:, :] = True
        border_mask[:, :8] = True
        border_mask[:, -8:] = True
        active_border_mask = border_mask & non_pad_mask
        center_mask = (~border_mask) & non_pad_mask

        border_grad_mean = float(grad_mag[active_border_mask].mean()) if np.any(active_border_mask) else 0.0
        center_grad_mean = float(grad_mag[center_mask].mean()) if np.any(center_mask) else 0.0
        border_to_center_grad_ratio = (
            border_grad_mean / center_grad_mean if center_grad_mean > 1e-6 else 0.0
        )

        # Cosine similarity to training HOG centroid of true class vs predicted class
        def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0

        sim_to_true_centroid = cos_sim(hog_vec, train_centroids[true_lbl])
        pred_lbl = rec["raw_predicted_label"]
        sim_to_pred_centroid = (
            cos_sim(hog_vec, train_centroids[pred_lbl]) if pred_lbl in train_centroids else 0.0
        )

        rec["reverified_raw_pred"] = pred_unflipped["predicted_label"]
        rec["true_label_rank_unflipped"] = int(true_rank_unflipped)
        rec["true_label_score_unflipped"] = float(true_score_unflipped)
        rec["counterfactual_flip_pred"] = pred_flipped["predicted_label"]
        rec["counterfactual_flip_correct"] = (pred_flipped["predicted_label"] == true_lbl)
        rec["true_label_rank_flipped"] = int(true_rank_flipped)
        rec["true_label_score_flipped"] = float(true_score_flipped)
        rec["white_pad_ratio"] = white_pad_ratio
        rec["border_grad_mean"] = border_grad_mean
        rec["center_grad_mean"] = center_grad_mean
        rec["border_to_center_grad_ratio"] = border_to_center_grad_ratio
        rec["sim_to_true_centroid"] = sim_to_true_centroid
        rec["sim_to_pred_centroid"] = sim_to_pred_centroid

    # Compute dataset reference border/center gradient stats on 200 train samples of A, B, C, D
    dataset_border_ratios: list[float] = []
    for lbl in ["A", "B", "C", "D"]:
        class_dir = PROJECT_ROOT / "data" / "raw" / "train" / lbl
        sample_files = sorted(class_dir.glob("*.jpg"))[:50]
        for img_p in sample_files:
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue
            _, _d_bgr64, g64 = extract_hog_with_diagnostics_from_bgr(bgr)
            gx = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g64.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.hypot(gx, gy)
            np_mask = g64 < 254
            b_mask = np.zeros_like(np_mask, dtype=bool)
            b_mask[:8, :] = True
            b_mask[-8:, :] = True
            b_mask[:, :8] = True
            b_mask[:, -8:] = True
            ab_mask = b_mask & np_mask
            c_mask = (~b_mask) & np_mask
            bg = float(gmag[ab_mask].mean()) if np.any(ab_mask) else 0.0
            cg = float(gmag[c_mask].mean()) if np.any(c_mask) else 0.0
            if cg > 1e-6:
                dataset_border_ratios.append(bg / cg)

    return {
        "dataset_mean_border_to_center_grad_ratio": float(np.mean(dataset_border_ratios)),
        "webcam_mean_border_to_center_grad_ratio": float(
            np.mean([r["border_to_center_grad_ratio"] for r in records])
        ),
    }


def generate_visualizations(records: list[dict]) -> None:
    wrong_records = [r for r in records if not r["is_correct_raw"]]

    # 1. Contact sheet of all misclassified samples
    n_wrong = len(wrong_records)
    cols = 4
    rows = max(1, math.ceil(n_wrong / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 3.3), dpi=140)
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.array(axes).reshape(-1))

    for idx, ax in enumerate(axes_flat):
        if idx >= n_wrong:
            ax.axis("off")
            continue
        rec = wrong_records[idx]
        crop_bgr = cv2.imread(str(rec["crop_path"]))
        gray_img = cv2.imread(str(rec["gray_path"]), cv2.IMREAD_GRAYSCALE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_disp = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        gray_disp = cv2.resize(gray_rgb, (128, 128), interpolation=cv2.INTER_NEAREST)
        sep = np.full((128, 6, 3), 240, dtype=np.uint8)
        combined = np.hstack([crop_disp, sep, gray_disp])

        ax.imshow(combined)
        ax.set_title(
            f"#{rec['stt']:02d} | Thật: {rec['verified_true_label']} -> Đoán: {rec['raw_predicted_label']} "
            f"(score={rec['predicted_decision_score']:.2f})\n"
            f"Hạng nhãn thật '{rec['verified_true_label']}': #{rec['true_label_rank_unflipped']} "
            f"({rec['true_label_score_unflipped']:.2f}) | {rec['folder_name'][:29]}...",
            fontsize=8,
            fontweight="bold",
            color="#991b1b",
        )
        ax.set_xticks([64, 134 + 64])
        ax.set_xticklabels(["Ảnh crop", "Ảnh xám 64×64"], fontsize=7.5)
        ax.set_yticks([])

    fig.suptitle(
        f"Chi tiết {n_wrong}/23 mẫu Webcam dự đoán sai (Nhãn thật verified_true_label vs Nhãn dự đoán thô)",
        fontsize=12.5,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(MISCLASSIFIED_PNG_PATH, bbox_inches="tight")
    plt.close(fig)

    # 2. Side-by-side comparison of representative webcam A, B, C, D vs training dataset A, B, C, D
    fig2, axes2 = plt.subplots(4, 3, figsize=(11, 11.5), dpi=140)
    signs = ["A", "B", "C", "D"]
    rep_map = {
        "A": next(r for r in records if r["verified_true_label"] == "A"),
        "B": next(r for r in records if r["verified_true_label"] == "B"),
        "C": next(r for r in records if r["verified_true_label"] == "C" and not r["is_correct_raw"]),
        "D": next(r for r in records if r["verified_true_label"] == "D" and not r["is_correct_raw"]),
    }
    for row_idx, sign in enumerate(signs):
        rec = rep_map[sign]
        w_gray = cv2.imread(str(rec["gray_path"]), cv2.IMREAD_GRAYSCALE)
        w_flip = cv2.flip(w_gray, 1)
        ds_sample_path = PROJECT_ROOT / "frontend" / "public" / "signs" / f"{sign}.jpg"
        ds_bgr = cv2.imread(str(ds_sample_path))
        _, _ds_bgr64, ds_gray = extract_hog_with_diagnostics_from_bgr(ds_bgr)

        axes2[row_idx, 0].imshow(w_gray, cmap="gray", vmin=0, vmax=255)
        axes2[row_idx, 0].set_title(
            f"Webcam #{rec['stt']:02d} (Thật: {sign})\nSVM đoán: {rec['raw_predicted_label']} "
            f"(Hạng '{sign}': #{rec['true_label_rank_unflipped']})",
            fontsize=9,
            fontweight="bold",
        )
        axes2[row_idx, 1].imshow(w_flip, cmap="gray", vmin=0, vmax=255)
        axes2[row_idx, 1].set_title(
            f"Webcam #{rec['stt']:02d} (Giả lập lật ngang)\nSVM đoán: {rec['counterfactual_flip_pred']} "
            f"(Hạng '{sign}': #{rec['true_label_rank_flipped']})",
            fontsize=9,
        )
        axes2[row_idx, 2].imshow(ds_gray, cmap="gray", vmin=0, vmax=255)
        axes2[row_idx, 2].set_title(
            f"Ảnh Dataset gốc lớp '{sign}' (64×64)\n(Nền phẳng đồng nhất, độ tương phản khác)",
            fontsize=9,
            color="#1e40af",
        )
        for c in range(3):
            axes2[row_idx, c].set_xticks([])
            axes2[row_idx, c].set_yticks([])

    fig2.suptitle(
        "So sánh ảnh xám 64×64 từ Webcam (Gốc & Lật ngang) với ảnh 64×64 trong Tập huấn luyện",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(COMPARISON_PNG_PATH, bbox_inches="tight")
    plt.close(fig2)


def main() -> int:
    records = load_confirmed_samples()
    predictor = SVMSignPredictor()
    extra_stats = analyze_with_predictor(records, predictor)
    generate_visualizations(records)

    # Compute per-class stats
    classes = ["A", "B", "C", "D"]
    per_class_stats = {}
    for cls in classes:
        cls_recs = [r for r in records if r["verified_true_label"] == cls]
        correct = sum(1 for r in cls_recs if r["is_correct_raw"])
        total = len(cls_recs)
        pred_counter = Counter(r["raw_predicted_label"] for r in cls_recs)
        per_class_stats[cls] = {
            "correct": correct,
            "total": total,
            "accuracy_pct": round(correct / total * 100.0, 2) if total else 0.0,
            "predicted_distribution": dict(pred_counter),
            "mean_true_label_rank": round(
                float(np.mean([r["true_label_rank_unflipped"] for r in cls_recs])), 2
            ),
            "mean_true_label_score": round(
                float(np.mean([r["true_label_score_unflipped"] for r in cls_recs])), 4
            ),
            "counterfactual_flip_correct": sum(
                1 for r in cls_recs if r["counterfactual_flip_correct"]
            ),
            "counterfactual_flip_distribution": dict(
                Counter(r["counterfactual_flip_pred"] for r in cls_recs)
            ),
        }

    total_correct = sum(1 for r in records if r["is_correct_raw"])
    total_samples = len(records)

    # Confusion matrix table: verified_true_label -> raw_predicted_label
    predicted_labels_seen = sorted(list({r["raw_predicted_label"] for r in records}))
    confusion_matrix: dict[str, dict[str, int]] = {}
    for cls in classes:
        cls_recs = [r for r in records if r["verified_true_label"] == cls]
        confusion_matrix[cls] = {
            pl: sum(1 for r in cls_recs if r["raw_predicted_label"] == pl)
            for pl in predicted_labels_seen
        }

    # Check saved flip_input_for_model distribution
    saved_flip_true = [r for r in records if r["flip_input_for_model"]]
    saved_flip_false = [r for r in records if not r["flip_input_for_model"]]

    summary = {
        "total_confirmed_samples": total_samples,
        "total_correct_raw": total_correct,
        "total_accuracy_pct": round(total_correct / total_samples * 100.0, 2),
        "per_class_stats": per_class_stats,
        "predicted_labels_seen": predicted_labels_seen,
        "confusion_matrix": confusion_matrix,
        "saved_flip_status": {
            "flip_true_count": len(saved_flip_true),
            "flip_false_count": len(saved_flip_false),
            "has_fair_paired_flip_samples": False,
            "counterfactual_flip_total_correct": sum(
                1 for r in records if r["counterfactual_flip_correct"]
            ),
        },
        "diagnostic_evidence": extra_stats,
        "samples": [
            {k: v for k, v in r.items() if k not in ("crop_path", "gray_path")}
            for r in records
        ],
    }

    REPORT_JSON_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Tổng số mẫu đã xác nhận: {total_samples}")
    print(f"Tổng số dự đoán đúng (raw): {total_correct}/{total_samples} ({summary['total_accuracy_pct']}%)")
    print("\n--- Thống kê theo từng lớp (verified_true_label) ---")
    for cls in classes:
        st = per_class_stats[cls]
        print(
            f"Lớp {cls}: {st['correct']}/{st['total']} ({st['accuracy_pct']}%) | "
            f"Phân bố nhãn đoán: {st['predicted_distribution']} | "
            f"Hạng TB của nhãn thật: #{st['mean_true_label_rank']} (score TB: {st['mean_true_label_score']}) | "
            f"Giả lập lật ngang đúng: {st['counterfactual_flip_correct']}/{st['total']} ({st['counterfactual_flip_distribution']})"
        )

    print("\n--- Trạng thái flip_input_for_model lúc lưu ---")
    print(
        f"flip_input_for_model=True: {len(saved_flip_true)} mẫu | "
        f"flip_input_for_model=False: {len(saved_flip_false)} mẫu"
    )

    print("\n--- Chi tiết từng mẫu ---")
    for r in records:
        status = "ĐÚNG" if r["is_correct_raw"] else "SAI "
        print(
            f"#{r['stt']:02d} [{status}] | Thật: {r['verified_true_label']} -> Đoán: {r['raw_predicted_label']} "
            f"(score={r['predicted_decision_score']:.3f}, margin={r['margin_top1_minus_top2']:.3f}) | "
            f"Hạng '{r['verified_true_label']}': #{r['true_label_rank_unflipped']} ({r['true_label_score_unflipped']:.3f}) | "
            f"Flip giả lập -> {r['counterfactual_flip_pred']} (Hạng '{r['verified_true_label']}': #{r['true_label_rank_flipped']}) | "
            f"Pad trắng: {r['white_pad_ratio']*100:.1f}% | {r['folder_name']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
