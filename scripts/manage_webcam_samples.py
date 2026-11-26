r"""List, visualize, and update/confirm ground-truth labels for saved webcam diagnostic samples.

Important constraint:
- This script intentionally does NOT compute accuracy statistics or train/tune any model
  while the user is still reviewing and correcting ground-truth labels.

Usage examples (PowerShell):
1. List all saved samples and regenerate the review CSV, Markdown, HTML, and contact sheet:
   .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py list

2. Update specific sample labels by index (#STT) or folder name:
   .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py set --updates "2=B,5=C,8=D"

3. Apply edits made directly in `results/svm/webcam_diagnostics/label_review_table.csv`:
   .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py sync-csv

4. Confirm all current labels as verified ground-truth:
   .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py confirm-all
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import hashlib


def export_collection_audit(samples, output):
    """Export collection metadata and near-duplicate warnings without changing labels or deleting images."""
    from collections import Counter, defaultdict
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fields = ['participant_id', 'session_id', 'gesture_instance_id',
              'handedness', 'background', 'lighting']
    rows, images, warnings, exact_duplicates = [], [], [], []
    seen_sha = {}
    for s in samples:
        row = {
            'folder_name': s['folder_name'],
            'true_label': s['current_label'],
            'label_confirmed': s['label_confirmed'],
        }
        row.update({key: str(s['raw_meta'].get(key, '') or '').strip() for key in fields})
        row['gesture_instance_key'] = (
            f"{row['participant_id']}|{row['session_id']}|{row['gesture_instance_id']}"
            if all(row[k] for k in ('participant_id', 'session_id', 'gesture_instance_id'))
            else ''
        )
        digest = hashlib.sha256(Path(s['crop_path']).read_bytes()).hexdigest()
        row['image_sha256'] = digest
        if digest in seen_sha:
            exact_duplicates.append({'first': seen_sha[digest], 'second': s['folder_name'], 'image_sha256': digest})
        else:
            seen_sha[digest] = s['folder_name']
        rows.append(row)
        image = cv2.imread(str(s['gray_path']), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            small = cv2.resize(image, (32, 32)).astype(np.float32)
            for name, previous in images:
                mae = float(np.mean(np.abs(previous - small)))
                if mae <= 8:
                    warnings.append({'first': name, 'second': s['folder_name'], 'gray_mae': mae})
            images.append((s['folder_name'], small))
    with (output / 'collection_metadata.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r['true_label'] for r in rows if r['label_confirmed'])
    all_counts = Counter(r['true_label'] for r in rows)
    instances_by_class = defaultdict(set)
    participants_by_class = defaultdict(set)
    for r in rows:
        if r['label_confirmed']:
            if r['gesture_instance_key']:
                instances_by_class[r['true_label']].add(r['gesture_instance_key'])
            if r['participant_id']:
                participants_by_class[r['true_label']].add(r['participant_id'])
    missing_confirmed_classes = [c for c in VALID_LABELS if counts[c] == 0]
    distinct_participants = sorted({r['participant_id'] for r in rows if r['participant_id']})
    report = {
        'total_samples': len(rows),
        'confirmed_samples': sum(1 for r in rows if r['label_confirmed']),
        'unconfirmed_samples': sum(1 for r in rows if not r['label_confirmed']),
        'confirmed_by_class': {c: counts[c] for c in VALID_LABELS},
        'all_by_class': {c: all_counts[c] for c in VALID_LABELS},
        'confirmed_classes_present_count': len(VALID_LABELS) - len(missing_confirmed_classes),
        'missing_confirmed_classes': missing_confirmed_classes,
        'distinct_participants': distinct_participants,
        'distinct_participants_by_class': {c: len(participants_by_class[c]) for c in VALID_LABELS},
        'distinct_gesture_instances_by_class': {c: len(instances_by_class[c]) for c in VALID_LABELS},
        'missing_metadata_samples': sum(any(not r[k] for k in fields) for r in rows),
        'by_participant_session': dict(Counter(str((r['participant_id'], r['session_id'])) for r in rows)),
        'exact_sha256_duplicate_warnings': exact_duplicates,
        'near_duplicate_warnings': warnings,
        'note': 'MAE <= 8 is a review heuristic, not proof of duplication. No images deleted or relabeled. Missing identity blocks a trustworthy grouped split.',
    }
    (output / 'collection_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

def summarize_collection_progress(
    samples: list[dict[str, object]],
    participant: str | None = None,
    target_instances: int = 5,
) -> dict[str, object]:
    """Summarize 36-class collection progress, distinct gesture instances, border-touch and near-duplicate warnings."""
    from collections import defaultdict

    pid_filter = str(participant or "").strip()
    filtered: list[dict[str, object]] = []
    for s in samples:
        raw_m = s["raw_meta"]
        p_val = str(raw_m.get("participant_id", "") or "").strip()
        if pid_filter and p_val != pid_filter:
            continue
        filtered.append(s)

    per_class_total: dict[str, int] = {c: 0 for c in VALID_LABELS}
    per_class_conf: dict[str, int] = {c: 0 for c in VALID_LABELS}
    per_class_unconf: dict[str, int] = {c: 0 for c in VALID_LABELS}
    per_class_instances: dict[str, set[str]] = defaultdict(set)
    border_touch_samples: list[dict[str, object]] = []

    seen_sha: dict[str, str] = {}
    exact_dups: list[dict[str, object]] = []
    near_dups: list[dict[str, object]] = []
    small_grays: list[tuple[str, np.ndarray]] = []

    for s in filtered:
        raw_m = s["raw_meta"]
        fname = str(s["folder_name"])
        lbl = str(s["current_label"]).strip().upper()
        conf = bool(s["label_confirmed"])
        pid = str(raw_m.get("participant_id", "") or "").strip()
        sid = str(raw_m.get("session_id", "") or "").strip()
        gid = str(raw_m.get("gesture_instance_id", "") or "").strip()

        if lbl in per_class_total:
            per_class_total[lbl] += 1
            if conf:
                per_class_conf[lbl] += 1
            else:
                per_class_unconf[lbl] += 1
            if pid and sid and gid:
                per_class_instances[lbl].add(f"{pid}|{sid}|{gid}")

        touches = raw_m.get("touches_frame_border")
        has_border_warn = bool(raw_m.get("border_touch_warning", False)) or (
            isinstance(touches, list) and len(touches) > 0
        )
        if has_border_warn:
            border_touch_samples.append(
                {
                    "folder_name": fname,
                    "label": lbl,
                    "gesture_instance_id": gid,
                    "touches_frame_border": touches if isinstance(touches, list) else [],
                }
            )

        crop_p = Path(s["crop_path"])  # type: ignore[arg-type]
        if crop_p.is_file():
            digest = hashlib.sha256(crop_p.read_bytes()).hexdigest()
            if digest in seen_sha:
                exact_dups.append({"first": seen_sha[digest], "second": fname, "image_sha256": digest})
            else:
                seen_sha[digest] = fname

        gray_p = Path(s["gray_path"])  # type: ignore[arg-type]
        if gray_p.is_file():
            img = cv2.imdecode(np.frombuffer(gray_p.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                small = cv2.resize(img, (32, 32)).astype(np.float32)
                for prev_name, prev_small in small_grays:
                    mae = float(np.mean(np.abs(prev_small - small)))
                    if mae <= 8.0:
                        near_dups.append({"first": prev_name, "second": fname, "gray_mae": round(mae, 4)})
                small_grays.append((fname, small))

    per_class: dict[str, dict[str, object]] = {}
    completed: list[str] = []
    in_progress: list[str] = []
    missing: list[str] = []
    for c in VALID_LABELS:
        inst_cnt = len(per_class_instances[c])
        rem = max(0, int(target_instances) - inst_cnt)
        if inst_cnt >= target_instances:
            status = "completed"
            completed.append(c)
        elif inst_cnt > 0 or per_class_total[c] > 0:
            status = "in_progress"
            in_progress.append(c)
        else:
            status = "missing"
            missing.append(c)
        per_class[c] = {
            "total_samples": per_class_total[c],
            "confirmed_samples": per_class_conf[c],
            "unconfirmed_samples": per_class_unconf[c],
            "distinct_gesture_instances": inst_cnt,
            "remaining_instances": rem,
            "status": status,
        }

    return {
        "participant_filter": pid_filter or "ALL",
        "target_instances_per_class": int(target_instances),
        "total_samples": len(filtered),
        "confirmed_samples": sum(1 for s in filtered if s["label_confirmed"]),
        "unconfirmed_samples": sum(1 for s in filtered if not s["label_confirmed"]),
        "classes_completed_count": len(completed),
        "classes_in_progress_count": len(in_progress),
        "classes_missing_count": len(missing),
        "completed_classes": completed,
        "in_progress_classes": in_progress,
        "missing_classes": missing,
        "border_touch_warning_count": len(border_touch_samples),
        "border_touch_warning_samples": border_touch_samples,
        "exact_sha256_duplicate_warning_count": len(exact_dups),
        "near_duplicate_warning_count": len(near_dups),
        "near_duplicate_warnings": near_dups,
        "per_class": per_class,
    }


def print_progress_report(report: dict[str, object], diag_dir: Path) -> None:
    target = int(report["target_instances_per_class"])
    print(f"\n================ TIẾN ĐỘ THU MẪU 36 LỚP ({diag_dir}) ================")
    print(f"Người chụp (--participant) : {report['participant_filter']}")
    print(f"Mục tiêu mỗi lớp           : {target} gesture_instance_id độc lập")
    print(
        f"Tổng mẫu                   : {report['total_samples']} "
        f"(Đã xác nhận: {report['confirmed_samples']} | Chờ xác nhận: {report['unconfirmed_samples']})"
    )
    print(
        f"Tiến độ lớp (trên 36 lớp)  : Đạt mục tiêu={report['classes_completed_count']}/36 | "
        f"Đang thu dở={report['classes_in_progress_count']} | Chưa có mẫu={report['classes_missing_count']}"
    )
    print(
        f"Cảnh báo chất lượng        : Chạm mép khung={report['border_touch_warning_count']} mẫu | "
        f"Gần trùng (MAE<=8)={report['near_duplicate_warning_count']} cặp | "
        f"Trùng SHA-256={report['exact_sha256_duplicate_warning_count']} cặp"
    )
    completed = report["completed_classes"]
    in_prog = report["in_progress_classes"]
    missing = report["missing_classes"]
    print(f"- Lớp ĐẠT mục tiêu ({len(completed)}): {', '.join(completed) if completed else 'Chưa có'}")
    if in_prog:
        per_c = report["per_class"]
        details = [
            f"{c} ({per_c[c]['distinct_gesture_instances']}/{target} lần, {per_c[c]['total_samples']} ảnh)"
            for c in in_prog
        ]
        print(f"- Lớp ĐANG THU DỞ ({len(in_prog)}): {', '.join(details)}")
    print(f"- Lớp CÒN THIẾU ({len(missing)}): {', '.join(missing) if missing else 'Không còn'}")
    print("=========================================================================")


DEFAULT_DIAG_DIR = Path("results/svm/webcam_diagnostics")
VALID_LABELS = [str(i) for i in range(10)] + [chr(ord("A") + i) for i in range(26)]


def load_saved_samples(diag_dir: Path = DEFAULT_DIAG_DIR) -> list[dict[str, object]]:
    if not diag_dir.is_dir():
        return []

    folders = sorted(
        p
        for p in diag_dir.iterdir()
        if p.is_dir() and (p / "diagnostic_metadata.json").is_file()
    )

    samples: list[dict[str, object]] = []
    for idx, folder in enumerate(folders, start=1):
        meta_path = folder / "diagnostic_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        crop_path = folder / "received_crop.png"
        gray_path = folder / "preprocessed_gray_64x64.png"

        original_label = str(
            meta.get("original_intended_sign") or meta.get("intended_sign") or ""
        ).strip()
        current_label = str(
            meta.get("verified_true_label") or meta.get("intended_sign") or ""
        ).strip()
        label_confirmed = bool(meta.get("label_confirmed", False))

        crop_b64 = (
            "data:image/png;base64,"
            + base64.b64encode(crop_path.read_bytes()).decode("ascii")
            if crop_path.is_file()
            else ""
        )
        gray_b64 = (
            "data:image/png;base64,"
            + base64.b64encode(gray_path.read_bytes()).decode("ascii")
            if gray_path.is_file()
            else ""
        )

        samples.append(
            {
                "stt": idx,
                "folder_name": folder.name,
                "folder_path": folder,
                "meta_path": meta_path,
                "crop_path": crop_path,
                "gray_path": gray_path,
                "original_label": original_label,
                "current_label": current_label,
                "label_confirmed": label_confirmed,
                "frame_id": str(meta.get("frame_id", "")),
                "saved_at": str(meta.get("saved_at", "")),
                "user_note": str(meta.get("user_note", "")),
                "flip_input_for_model": bool(meta.get("flip_input_for_model", False)),
                "crop_data_url": crop_b64,
                "gray_data_url": gray_b64,
                "raw_meta": meta,
            }
        )
    return samples


def update_sample_label(
    sample: dict[str, object],
    new_label: str,
    confirmed: bool = True,
    note: str | None = None,
) -> None:
    clean_lbl = new_label.strip().upper()
    if clean_lbl not in VALID_LABELS:
        raise ValueError(
            f"Nhãn '{new_label}' không thuộc danh sách 36 lớp hợp lệ (0–9, A–Z)."
        )

    meta_path = Path(sample["meta_path"])  # type: ignore[arg-type]
    meta = dict(sample["raw_meta"])  # type: ignore[arg-type]

    if "original_intended_sign" not in meta:
        meta["original_intended_sign"] = meta.get("intended_sign", "")

    meta["intended_sign"] = clean_lbl
    meta["verified_true_label"] = clean_lbl
    meta["label_confirmed"] = bool(confirmed)
    meta["label_updated_at"] = datetime.now().isoformat()
    if note is not None:
        meta["user_note"] = note

    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def export_review_artifacts(
    samples: list[dict[str, object]],
    diag_dir: Path = DEFAULT_DIAG_DIR,
) -> dict[str, Path]:
    diag_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diag_dir / "label_review_table.csv"
    md_path = diag_dir / "label_review_table.md"
    sheet_path = diag_dir / "review_contact_sheet.png"
    html_path = diag_dir / "review_samples.html"

    # 1. Write CSV review table
    csv_fields = [
        "stt",
        "folder_name",
        "current_label",
        "confirmed_true_label",
        "label_confirmed",
        "original_label",
        "participant_id",
        "session_id",
        "gesture_instance_id",
        "handedness",
        "background",
        "lighting",
        "frame_id",
        "saved_at",
        "user_note",
        "crop_image_path",
        "gray_64x64_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for s in samples:
            raw_m = s["raw_meta"]
            writer.writerow(
                {
                    "stt": s["stt"],
                    "folder_name": s["folder_name"],
                    "current_label": s["current_label"],
                    "confirmed_true_label": s["current_label"],
                    "label_confirmed": s["label_confirmed"],
                    "original_label": s["original_label"],
                    "participant_id": str(raw_m.get("participant_id", "") or ""),
                    "session_id": str(raw_m.get("session_id", "") or ""),
                    "gesture_instance_id": str(raw_m.get("gesture_instance_id", "") or ""),
                    "handedness": str(raw_m.get("handedness", "") or ""),
                    "background": str(raw_m.get("background", "") or ""),
                    "lighting": str(raw_m.get("lighting", "") or ""),
                    "frame_id": s["frame_id"],
                    "saved_at": s["saved_at"],
                    "user_note": s["user_note"],
                    "crop_image_path": str(Path(s["crop_path"]).as_posix()),  # type: ignore[arg-type]
                    "gray_64x64_path": str(Path(s["gray_path"]).as_posix()),  # type: ignore[arg-type]
                }
            )

    # 2. Write Markdown review table
    md_lines = [
        "# Bảng Rà soát Nhãn thật các Mẫu Webcam Đã Lưu",
        "",
        "> **Lưu ý**: Bảng này chỉ hiển thị ảnh đã lưu, số thứ tự (`#STT`), tên thư mục và nhãn hiện tại để bạn kiểm tra và xác nhận/sửa nhãn thật. **Chưa thống kê độ chính xác và chưa dùng các mẫu này để huấn luyện.**",
        "",
        "| STT | Tên thư mục | Nhãn hiện tại | Đã xác nhận? | Ghi chú | Ảnh crop màu | Ảnh xám 64×64 |",
        "| :---: | :--- | :---: | :---: | :--- | :---: | :---: |",
    ]
    for s in samples:
        crop_rel = f"{s['folder_name']}/received_crop.png"
        gray_rel = f"{s['folder_name']}/preprocessed_gray_64x64.png"
        conf_txt = "Đã xác nhận" if s["label_confirmed"] else "Chờ xác nhận"
        note_txt = str(s["user_note"] or "—")
        md_lines.append(
            f"| **#{int(s['stt']):02d}** | `{s['folder_name']}` | **`{s['current_label']}`** | "
            f"{conf_txt} | {note_txt} | [received_crop.png]({crop_rel}) | [gray_64x64.png]({gray_rel}) |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # 3. Generate visual contact sheet PNG
    n = len(samples)
    if n > 0:
        cols = 4
        grid_rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            grid_rows,
            cols,
            figsize=(16, max(3.8, 3.7 * grid_rows)),
            dpi=140,
            squeeze=False,
        )
        for i in range(grid_rows * cols):
            ax = axes[i // cols][i % cols]
            if i >= n:
                ax.axis("off")
                continue
            s = samples[i]
            crop_bgr = cv2.imread(str(s["crop_path"]))
            gray_bgr = cv2.imread(str(s["gray_path"]))

            canvas = np.full((140, 284, 3), 244, dtype=np.uint8)
            if crop_bgr is not None:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                ch, cw = crop_rgb.shape[:2]
                scale = min(132.0 / cw, 132.0 / ch)
                nw = max(1, int(round(cw * scale)))
                nh = max(1, int(round(ch * scale)))
                c_res = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                y0 = (140 - nh) // 2
                x0 = (138 - nw) // 2
                canvas[y0 : y0 + nh, x0 : x0 + nw] = c_res

            if gray_bgr is not None:
                gray_rgb = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGB)
                g_res = cv2.resize(gray_rgb, (132, 132), interpolation=cv2.INTER_NEAREST)
                canvas[4:136, 148:280] = g_res

            ax.imshow(canvas)
            status_tag = "✓ Đã xác nhận" if s["label_confirmed"] else "Chờ xác nhận"
            note_part = f" | {s['user_note']}" if s["user_note"] else ""
            title_line1 = f"#{int(s['stt']):02d} — Nhãn hiện tại: '{s['current_label']}' ({status_tag}){note_part}"
            title_line2 = str(s["folder_name"])
            ax.set_title(
                f"{title_line1}\n{title_line2}",
                fontsize=7.8,
                fontweight="bold",
            )
            ax.set_xticks([69, 214])
            ax.set_xticklabels(["Ảnh crop màu", "Ảnh xám 64×64"], fontsize=7.5)
            ax.set_yticks([])

        fig.suptitle(
            f"Bảng xem {n} mẫu Webcam đã lưu (Kèm #STT, tên thư mục và nhãn hiện tại — Chưa thống kê độ chính xác)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(sheet_path)
        plt.close(fig)

    # 4. Generate standalone interactive HTML review page
    html_rows = []
    options_template = VALID_LABELS
    for s in samples:
        stt = int(s["stt"])
        curr = str(s["current_label"])
        folder_esc = html.escape(str(s["folder_name"]))
        note_esc = html.escape(str(s["user_note"]))
        saved_esc = html.escape(str(s["saved_at"]))
        opts = []
        for lbl in options_template:
            sel = " selected" if lbl == curr else ""
            opts.append(f'<option value="{lbl}"{sel}>{lbl}</option>')
        opts_html = "".join(opts)
        badge_cls = "confirmed" if s["label_confirmed"] else "pending"
        badge_txt = "Đã xác nhận" if s["label_confirmed"] else "Chờ xác nhận"
        html_rows.append(
            f"""
            <tr data-stt="{stt}" data-orig="{html.escape(curr)}">
              <td class="stt">#{stt:02d}</td>
              <td>
                <div class="folder">{folder_esc}</div>
                <div class="meta">Frame: {html.escape(str(s['frame_id']))} · Lưu lúc: {saved_esc}</div>
                <div class="meta">Ghi chú: {note_esc or '—'}</div>
              </td>
              <td class="img-cell"><img src="{s['crop_data_url']}" alt="Crop #{stt:02d}" /></td>
              <td class="img-cell"><img class="pixelated" src="{s['gray_data_url']}" alt="Gray 64x64 #{stt:02d}" /></td>
              <td class="curr-label"><strong>{html.escape(curr)}</strong><br/><span class="badge {badge_cls}">{badge_txt}</span></td>
              <td>
                <select class="label-select" onchange="updateCommand()">
                  {opts_html}
                </select>
              </td>
            </tr>
            """
        )

    diag_dir_flag = ""
    if diag_dir.resolve() != DEFAULT_DIAG_DIR.resolve():
        diag_dir_flag = f' --diag-dir "{diag_dir.as_posix()}"'
    default_cmd = f".\\.venv\\Scripts\\python.exe scripts/manage_webcam_samples.py confirm-all{diag_dir_flag}"
    html_doc = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>Bảng rà soát & xác nhận nhãn thật các mẫu Webcam</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
    h1 {{ font-size: 20px; margin-bottom: 6px; }}
    p.note {{ font-size: 13px; color: #475569; margin-top: 0; }}
    .cmd-box {{ background: #0f172a; color: #e2e8f0; padding: 14px 16px; border-radius: 8px; margin-bottom: 18px; font-family: Consolas, monospace; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(15,23,42,0.06); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: middle; font-size: 13px; }}
    th {{ background: #f1f5f9; font-weight: 600; }}
    td.stt {{ font-weight: 700; font-size: 15px; width: 60px; color: #2563eb; }}
    .folder {{ font-family: Consolas, monospace; font-weight: 600; font-size: 12.5px; }}
    .meta {{ font-size: 11.5px; color: #64748b; margin-top: 3px; }}
    td.img-cell {{ width: 120px; text-align: center; }}
    td.img-cell img {{ width: 96px; height: 96px; object-fit: contain; background: #1e293b; border-radius: 6px; border: 1px solid #cbd5e1; }}
    td.img-cell img.pixelated {{ image-rendering: pixelated; background: #fff; }}
    td.curr-label {{ text-align: center; font-size: 16px; width: 120px; }}
    .badge {{ display: inline-block; margin-top: 4px; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
    .badge.confirmed {{ background: #dcfce7; color: #166534; }}
    .badge.pending {{ background: #fef3c7; color: #92400e; }}
    select.label-select {{ padding: 6px 10px; font-size: 14px; font-weight: 600; border-radius: 6px; border: 1px solid #94a3b8; }}
    tr.changed {{ background: #eff6ff; }}
  </style>
</head>
<body>
  <h1>Bảng xem và xác nhận nhãn thật cho {n} mẫu Webcam đã lưu</h1>
  <p class="note">Chưa thống kê độ chính xác và chưa dùng các mẫu để huấn luyện. Hãy chọn lại nhãn thật ở cột cuối cho các mẫu bị nhầm, sau đó sao chép lệnh PowerShell bên dưới để cập nhật.</p>
  <div class="cmd-box" id="cmdOutput">
    {html.escape(default_cmd)}
  </div>
  <table>
    <thead>
      <tr>
        <th>STT</th>
        <th>Tên thư mục & Thông tin</th>
        <th>Ảnh crop màu</th>
        <th>Ảnh xám 64×64</th>
        <th>Nhãn hiện tại</th>
        <th>Chọn / Sửa nhãn thật</th>
      </tr>
    </thead>
    <tbody>
      {''.join(html_rows)}
    </tbody>
  </table>
  <script>
    const diagDirFlag = {json.dumps(diag_dir_flag)};
    function updateCommand() {{
      const rows = document.querySelectorAll('tbody tr');
      const updates = [];
      rows.forEach(tr => {{
        const stt = tr.getAttribute('data-stt');
        const orig = tr.getAttribute('data-orig');
        const sel = tr.querySelector('select').value;
        if (sel !== orig) {{
          tr.classList.add('changed');
          updates.push(stt + '=' + sel);
        }} else {{
          tr.classList.remove('changed');
        }}
      }});
      const box = document.getElementById('cmdOutput');
      if (updates.length > 0) {{
        box.textContent = '.\\\\.venv\\\\Scripts\\\\python.exe scripts/manage_webcam_samples.py set' + diagDirFlag + ' --updates "' + updates.join(',') + '" --confirm-all';
      }} else {{
        box.textContent = '.\\\\.venv\\\\Scripts\\\\python.exe scripts/manage_webcam_samples.py confirm-all' + diagDirFlag;
      }}
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")

    return {
        "csv": csv_path,
        "md": md_path,
        "contact_sheet": sheet_path,
        "html": html_path,
    }


def parse_updates_arg(updates_str: str) -> dict[str, str]:
    """Parse '2=B, 5=C, sample_xyz=D' into a dict mapping key -> new_label."""
    mapping: dict[str, str] = {}
    for part in updates_str.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Định dạng cập nhật không hợp lệ: '{item}'. Hãy dùng dạng 'STT=NHÃN' (ví dụ: '3=B,7=C')."
            )
        k, v = item.split("=", 1)
        mapping[k.strip().lstrip("#")] = v.strip().upper()
    return mapping


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Xem, xác nhận hoặc sửa nhãn thật của các mẫu webcam đã lưu."
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "set", "sync-csv", "confirm-all", "progress"],
        help="Hành động cần thực hiện (mặc định: list)",
    )
    parser.add_argument(
        "--diag-dir",
        type=Path,
        default=DEFAULT_DIAG_DIR,
        help=f"Thư mục chứa các mẫu webcam đã lưu (mặc định: {DEFAULT_DIAG_DIR})",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="",
        help="Lọc theo participant_id khi dùng action 'progress' (VD: P01; bỏ trống = tất cả)",
    )
    parser.add_argument(
        "--target-instances",
        type=int,
        default=5,
        help="Số gesture_instance_id độc lập mục tiêu cho mỗi lớp khi dùng 'progress' (mặc định: 5)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Tùy chọn lưu báo cáo JSON khi dùng action 'progress'",
    )
    parser.add_argument(
        "--updates",
        type=str,
        default="",
        help="Chuỗi cập nhật nhãn theo STT hoặc tên thư mục, ví dụ: '2=B,5=C,14=D'",
    )
    parser.add_argument(
        "--confirm-all",
        action="store_true",
        help="Đánh dấu tất cả các mẫu là đã xác nhận nhãn thật sau khi cập nhật",
    )
    parser.add_argument('--collection-audit', type=Path, help='New output directory for metadata CSV and duplicate audit; does not update labels')
    parser.add_argument('--edit-metadata', type=Path, help='JSON: folder_name -> six collection fields; labels untouched')
    args = parser.parse_args(argv)

    samples = load_saved_samples(args.diag_dir)
    if args.action == "progress":
        report = summarize_collection_progress(
            samples,
            participant=args.participant,
            target_instances=args.target_instances,
        )
        print_progress_report(report, args.diag_dir)
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Đã lưu báo cáo tiến độ JSON tại: {args.json_output.resolve()}")
        return 0

    if not samples:
        print(f"Không tìm thấy mẫu webcam nào trong: {args.diag_dir}")
        return 0

    if args.edit_metadata:
        updates = json.loads(args.edit_metadata.read_text(encoding='utf-8'))
        allowed = {'participant_id', 'session_id', 'gesture_instance_id', 'handedness', 'background', 'lighting'}
        targets = {s['folder_name']: s for s in samples}
        for name, values in updates.items():
            if name not in targets or not isinstance(values, dict) or set(values) - allowed:
                raise ValueError('Unknown sample or invalid metadata fields: ' + name)
            if any(not isinstance(v, str) or not v.strip() for v in values.values()):
                raise ValueError('Metadata values must be nonempty strings')
            if 'handedness' in values and values['handedness'] not in ('left', 'right'):
                raise ValueError('handedness must be left or right')
        for name, values in updates.items():
            s = targets[name]
            meta = dict(s['raw_meta'])
            meta.update(values)
            Path(s['meta_path']).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        return 0

    if args.collection_audit:
        export_collection_audit(samples, args.collection_audit)
        return 0

    if args.action == "set":
        if not args.updates.strip():
            print("Lỗi: Vui lòng truyền --updates 'STT=NHÃN,...' khi dùng lệnh set.", file=sys.stderr)
            return 1
        updates_map = parse_updates_arg(args.updates)
        by_stt = {str(int(s["stt"])): s for s in samples}
        by_stt_padded = {f"{int(s['stt']):02d}": s for s in samples}
        by_folder = {str(s["folder_name"]): s for s in samples}

        updated_count = 0
        for key, new_lbl in updates_map.items():
            target = by_stt.get(key) or by_stt_padded.get(key) or by_folder.get(key)
            if target is None:
                print(f"Lỗi: Không tìm thấy mẫu có STT hoặc tên thư mục '{key}'.", file=sys.stderr)
                return 1
            old_lbl = target["current_label"]
            update_sample_label(target, new_lbl, confirmed=True)
            updated_count += 1
            print(
                f"Đã sửa mẫu #{int(target['stt']):02d} ({target['folder_name']}): "
                f"'{old_lbl}' -> '{new_lbl}'"
            )

        if args.confirm_all:
            samples = load_saved_samples(args.diag_dir)
            for s in samples:
                update_sample_label(s, str(s["current_label"]), confirmed=True)
            print(f"Đã đánh dấu xác nhận toàn bộ {len(samples)} mẫu.")

        samples = load_saved_samples(args.diag_dir)

    elif args.action == "sync-csv":
        csv_path = args.diag_dir / "label_review_table.csv"
        if not csv_path.is_file():
            print(f"Lỗi: Không tìm thấy file CSV tại {csv_path}", file=sys.stderr)
            return 1
        by_folder = {str(s["folder_name"]): s for s in samples}
        changed = 0
        with csv_path.open("r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = row.get("folder_name", "").strip()
                new_lbl = (
                    row.get("confirmed_true_label") or row.get("current_label") or ""
                ).strip().upper()
                if fname in by_folder and new_lbl:
                    s = by_folder[fname]
                    old_lbl = str(s["current_label"])
                    note = row.get("user_note", str(s["user_note"]))
                    update_sample_label(s, new_lbl, confirmed=True, note=note)
                    if new_lbl != old_lbl:
                        changed += 1
                        print(f"Đồng bộ CSV mẫu #{int(s['stt']):02d} ({fname}): '{old_lbl}' -> '{new_lbl}'")
        print(f"Đã đồng bộ xong từ CSV ({changed} mẫu đổi nhãn, toàn bộ {len(samples)} mẫu đã xác nhận).")
        samples = load_saved_samples(args.diag_dir)

    elif args.action == "confirm-all":
        for s in samples:
            update_sample_label(s, str(s["current_label"]), confirmed=True)
        print(f"Đã xác nhận nhãn thật cho toàn bộ {len(samples)} mẫu.")
        samples = load_saved_samples(args.diag_dir)

    artifacts = export_review_artifacts(samples, args.diag_dir)

    print(f"\n================ DANH SÁCH {len(samples)} MẪU WEBCAM ĐÃ LƯU ================")
    for s in samples:
        status = "ĐÃ XÁC NHẬN" if s["label_confirmed"] else "Chờ xác nhận"
        note_str = f" | Ghi chú: {s['user_note']}" if s["user_note"] else ""
        print(
            f"#{int(s['stt']):02d} | {s['folder_name']} | "
            f"Nhãn hiện tại: {s['current_label']} ({status}){note_str}"
        )
    print("=========================================================================")
    print(f"- Bảng ảnh tổng hợp (PNG) : {artifacts['contact_sheet'].resolve()}")
    print(f"- Bảng rà soát (CSV)      : {artifacts['csv'].resolve()}")
    print(f"- Bảng rà soát (Markdown) : {artifacts['md'].resolve()}")
    print(f"- Trang xem & chọn nhãn   : {artifacts['html'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
