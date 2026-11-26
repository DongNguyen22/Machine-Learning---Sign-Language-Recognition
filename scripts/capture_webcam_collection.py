"""Webcam capture tool for SignVision (36 classes: 0-9, A-Z) with collection metadata.

Inspired by the structured capture step of computervisioneng/sign-language-detector-python
(`collect_imgs.py`), adapted specifically for SignVision's LinearSVC + HOG workflow:
- Uses MediaPipe HandLandmarker (with exact same largest-hand + 18% rectangular padded crop
  as `frontend/src/recognition/predictionLogic.ts`) by default, with manual ROI fallback.
- Computes `preprocessed_gray_64x64.png`, `legacy_gray_64x64.png`, and `method_3b_gray_64x64.png`
  strictly through the shared functions in `src/preprocess/build_hog_dataset.py` so saved
  samples match live inference bit-for-bit.
- Enforces the 6 required collection metadata fields (`participant_id`, `session_id`,
  `gesture_instance_id`, `handedness`, `background`, `lighting`) and keeps
  `label_confirmed=False` until explicit human review via `scripts/manage_webcam_samples.py`.
- Warns immediately on exact/near-duplicate images (SHA-256 or 32x32 gray MAE <= 8.0)
  WITHOUT deleting or auto-relabeling any image.
- Encourages lowering and re-raising the hand between distinct gesture instances (press `N`
  to increment `gesture_instance_id`, or `SPACE` to capture & review, `ESC`/`Q` to quit).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from uuid import uuid4

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.diagnostic_storage import (
    detect_crop_border_touches,
    suggest_next_gesture_instance_id,
)
from src.preprocess.build_hog_dataset import (
    apply_webcam_method_3b_from_bgr,
    extract_hog_with_diagnostics_from_bgr,
)

VALID_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
REQUIRED_METADATA_FIELDS = (
    "participant_id",
    "session_id",
    "gesture_instance_id",
    "handedness",
    "background",
    "lighting",
)
DEFAULT_HAND_LANDMARKER_MODEL = PROJECT_ROOT / "results" / "svm" / "hand_landmarker.task"
NEAR_DUPLICATE_MAE_THRESHOLD = 8.0


def compute_padded_hand_crop_box(
    landmarks,
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.18,
) -> dict[str, int] | None:
    """Match `frontend/src/recognition/predictionLogic.ts::computePaddedHandCropBox`."""
    if not landmarks or frame_width <= 1 or frame_height <= 1:
        return None
    xs = [float(p.x) * frame_width for p in landmarks]
    ys = [float(p.y) * frame_height for p in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    raw_w = max(16.0, max_x - min_x)
    raw_h = max(16.0, max_y - min_y)
    safe_ratio = max(0.05, float(padding_ratio))
    min_pad_px = max(12, round(min(frame_width, frame_height) * 0.045))
    pad_x = max(min_pad_px, round(raw_w * safe_ratio))
    pad_y = max(min_pad_px, round(raw_h * safe_ratio))

    x1 = max(0, math.floor(min_x - pad_x))
    y1 = max(0, math.floor(min_y - pad_y))
    x2 = min(frame_width, math.ceil(max_x + pad_x))
    y2 = min(frame_height, math.ceil(max_y + pad_y))
    return {
        "x": int(x1),
        "y": int(y1),
        "width": int(max(1, x2 - x1)),
        "height": int(max(1, y2 - y1)),
    }


def select_largest_hand_landmarks(hand_landmarks_list, frame_width: int, frame_height: int):
    """Select the single hand with largest bounding-box pixel area."""
    if not hand_landmarks_list:
        return None, -1
    best_idx = 0
    best_area = -1.0
    for idx, hand in enumerate(hand_landmarks_list):
        if not hand:
            continue
        xs = [float(p.x) * frame_width for p in hand]
        ys = [float(p.y) * frame_height for p in hand]
        area = max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best_idx = idx
    return hand_landmarks_list[best_idx], best_idx


def increment_instance_id(instance_id: str) -> str:
    """Increment trailing digits in gesture_instance_id (e.g. G01 -> G02, inst1 -> inst2)."""
    match = re.search(r"^(.*?)(\d+)$", instance_id.strip())
    if not match:
        return f"{instance_id.strip()}_2"
    prefix, digits = match.group(1), match.group(2)
    next_num = int(digits) + 1
    return f"{prefix}{next_num:0{len(digits)}d}"


def parse_labels_spec(label: str | None, labels_spec: str | None) -> list[str]:
    """Parse either legacy single `--label` or batch `--labels` (comma list, ranges `0-9`/`A-M`, or `ALL`)."""
    has_single = bool(label and label.strip())
    has_multi = bool(labels_spec and labels_spec.strip())
    if has_single and has_multi:
        raise ValueError("Chỉ chọn một trong hai: --label (1 lớp) hoặc --labels (nhiều lớp tuần tự).")
    if not has_single and not has_multi:
        raise ValueError("Cần truyền --label (VD: A) hoặc --labels (VD: 0-9, A-M, N-Z, ALL).")

    if has_single:
        clean = str(label).strip().upper()
        if clean not in VALID_LABELS:
            raise ValueError(f"Nhãn không hợp lệ: {label!r}. Phải thuộc 0–9 hoặc A–Z.")
        return [clean]

    raw_spec = str(labels_spec).strip().upper()
    result: list[str] = []
    seen: set[str] = set()

    def add_lbl(lbl: str) -> None:
        if lbl not in VALID_LABELS:
            raise ValueError(f"Nhãn không hợp lệ trong --labels: {lbl!r}.")
        if lbl not in seen:
            seen.add(lbl)
            result.append(lbl)

    for token in (t.strip() for t in raw_spec.split(",") if t.strip()):
        if token == "ALL":
            for lbl in VALID_LABELS:
                add_lbl(lbl)
        elif "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or len(parts[0]) != 1 or len(parts[1]) != 1:
                raise ValueError(f"Khoảng nhãn không hợp lệ: {token!r} (dùng dạng 0-9, A-M, N-Z).")
            start_c, end_c = parts[0], parts[1]
            if start_c.isdigit() and end_c.isdigit():
                s_i, e_i = int(start_c), int(end_c)
                if s_i > e_i:
                    raise ValueError(f"Khoảng số tăng dần không hợp lệ: {token!r}.")
                for n in range(s_i, e_i + 1):
                    add_lbl(str(n))
            elif start_c.isalpha() and end_c.isalpha():
                s_o, e_o = ord(start_c), ord(end_c)
                if s_o > e_o:
                    raise ValueError(f"Khoảng chữ tăng dần không hợp lệ: {token!r}.")
                for code in range(s_o, e_o + 1):
                    add_lbl(chr(code))
            else:
                raise ValueError(f"Khoảng nhãn không cùng loại số/chữ: {token!r}.")
        else:
            add_lbl(token)

    if not result:
        raise ValueError("Danh sách --labels rỗng.")
    return result


def check_near_duplicates(
    root: Path,
    crop_png_bytes: bytes,
    gray_64: np.ndarray,
    mae_threshold: float = NEAR_DUPLICATE_MAE_THRESHOLD,
) -> list[dict[str, object]]:
    """Compare candidate image against existing saved samples (warning only, never deletes)."""
    warnings: list[dict[str, object]] = []
    if not root.is_dir():
        return warnings

    candidate_sha = hashlib.sha256(crop_png_bytes).hexdigest()
    candidate_small = cv2.resize(gray_64, (32, 32)).astype(np.float32)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / "diagnostic_metadata.json").is_file():
            continue
        existing_crop = folder / "received_crop.png"
        existing_gray = folder / "preprocessed_gray_64x64.png"
        exact_dup = False
        if existing_crop.is_file():
            if hashlib.sha256(existing_crop.read_bytes()).hexdigest() == candidate_sha:
                exact_dup = True
        mae = None
        if existing_gray.is_file():
            prev_gray = cv2.imread(str(existing_gray), cv2.IMREAD_GRAYSCALE)
            if prev_gray is not None:
                prev_small = cv2.resize(prev_gray, (32, 32)).astype(np.float32)
                mae = float(np.mean(np.abs(prev_small - candidate_small)))
        if exact_dup or (mae is not None and mae <= mae_threshold):
            warnings.append(
                {
                    "existing_folder": folder.name,
                    "exact_sha256_match": exact_dup,
                    "gray_mae": round(mae, 4) if mae is not None else None,
                }
            )
    return warnings


def save_sample(
    root: Path | str,
    image: np.ndarray,
    metadata: dict[str, object],
    roi_mode: str = "manual",
    crop_box: dict[str, int] | None = None,
    source_frame_dimensions: dict[str, int] | None = None,
) -> Path:
    """Atomically save a captured BGR hand crop with shared HOG 64x64 preprocessing and metadata."""
    if image is None or not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a non-empty 3-channel BGR numpy array.")

    for key in REQUIRED_METADATA_FIELDS:
        val = metadata.get(key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Missing or empty metadata field: {key}")
    if metadata["handedness"] not in ("left", "right"):
        raise ValueError("handedness must be 'left' or 'right'")
    intended = str(metadata.get("intended_sign", "")).strip().upper()
    if intended not in VALID_LABELS:
        raise ValueError(f"Invalid intended_sign: {intended}")

    legacy_diag = extract_hog_with_diagnostics_from_bgr(image)
    if legacy_diag is None:
        raise ValueError("Cannot extract legacy 64x64 grayscale image from BGR crop.")
    _, _, legacy_gray_64 = legacy_diag

    m3b_bgr, _, _ = apply_webcam_method_3b_from_bgr(image, already_flipped=False)
    m3b_diag = extract_hog_with_diagnostics_from_bgr(m3b_bgr)
    if m3b_diag is None:
        raise ValueError("Cannot extract Method 3B 64x64 grayscale image from BGR crop.")
    _, _, m3b_gray_64 = m3b_diag

    ok_crop, crop_buf = cv2.imencode(".png", image)
    ok_leg, leg_buf = cv2.imencode(".png", legacy_gray_64)
    ok_m3b, m3b_buf = cv2.imencode(".png", m3b_gray_64)
    if not (ok_crop and ok_leg and ok_m3b):
        raise IOError("Cannot PNG-encode captured crop or 64x64 gray images.")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    near_dups = check_near_duplicates(root_path, crop_buf.tobytes(), legacy_gray_64)

    token = uuid4().hex
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_id = f"CLI_{ts}"
    staging = root_path / f".pending_{token}"
    folder = root_path / f"sample_{token}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "received_crop.png").write_bytes(crop_buf.tobytes())
        (staging / "preprocessed_gray_64x64.png").write_bytes(leg_buf.tobytes())
        (staging / "legacy_gray_64x64.png").write_bytes(leg_buf.tobytes())
        (staging / "method_3b_gray_64x64.png").write_bytes(m3b_buf.tobytes())

        touches_frame_border = detect_crop_border_touches(crop_box, source_frame_dimensions)
        meta = dict(metadata)
        meta.update(
            {
                "intended_sign": intended,
                "saved_at": datetime.now().isoformat(),
                "frame_id": frame_id,
                "image_frame_id": frame_id,
                "label_confirmed": False,
                "flip_input_for_model": False,
                "preview_mirrored": False,
                "experimental_webcam_mode": False,
                "webcam_mode": "legacy",
                "roi_mode": roi_mode,
                "crop_box": crop_box,
                "source_frame_dimensions": source_frame_dimensions,
                "touches_frame_border": touches_frame_border,
                "border_touch_warning": len(touches_frame_border) > 0,
                "crop_orientation": "canonical_unflipped_raw_camera",
                "gray_source": "shared_extract_hog_with_diagnostics_from_bgr",
                "method_3b_flip_count_from_saved_crop": 1,
                "collection_pipeline": f"cli_{roi_mode}_unmirrored_v2",
                "near_duplicate_warnings_at_capture": near_dups,
            }
        )
        (staging / "diagnostic_metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        staging.rename(folder)
    except Exception:
        for p in staging.glob("*"):
            p.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return folder


def open_camera_with_fallback(camera_index: int) -> tuple[cv2.VideoCapture, str]:
    """Open webcam and verify at least one readable frame, falling back to DirectShow on Windows."""
    backends: list[tuple[str, int | None]] = [("default", None)]
    if sys.platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        backends.append(("CAP_DSHOW", cv2.CAP_DSHOW))

    for backend_name, backend_flag in backends:
        cap = (
            cv2.VideoCapture(camera_index)
            if backend_flag is None
            else cv2.VideoCapture(camera_index, backend_flag)
        )
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, backend_name
        cap.release()

    raise RuntimeError(
        f"Không thể mở hoặc đọc khung hình từ camera index={camera_index} "
        f"(đã thử: {', '.join(name for name, _ in backends)})."
    )


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True, help="Output folder for captured samples")
    for field in ("participant_id", "session_id", "background", "lighting"):
        p.add_argument("--" + field.replace("_", "-"), required=True)
    p.add_argument(
        "--gesture-instance-id",
        default="auto",
        help="Mã lần giơ tay (VD: A_G01 hoặc 'auto' để tự gợi ý mã chưa dùng; mặc định: auto)",
    )
    p.add_argument("--handedness", choices=["left", "right"], required=True)
    p.add_argument("--label", choices=VALID_LABELS, default=None, help="Thu 1 lớp đơn (tương thích ngược)")
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Thu tuần tự nhiều lớp trong 1 phiên mở camera (VD: '0-9', 'A-M', 'N-Z', 'ALL' hoặc '0,1,A,B')",
    )
    p.add_argument(
        "--roi-mode",
        choices=["mediapipe", "manual"],
        default="mediapipe",
        help="Use MediaPipe 18%% padded hand crop (matches SignVision UI) or manual ROI selection",
    )
    p.add_argument(
        "--hand-model",
        type=Path,
        default=DEFAULT_HAND_LANDMARKER_MODEL,
        help=f"Path to MediaPipe hand_landmarker.task (default: {DEFAULT_HAND_LANDMARKER_MODEL})",
    )
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    p.add_argument(
        "--count",
        type=int,
        default=1,
        help="Số mẫu cần lưu cho MỖI lớp trước khi chuyển lớp kế tiếp/thoát (mặc định: 1)",
    )
    p.add_argument(
        "--auto-next-instance",
        action="store_true",
        help="Automatically increment gesture_instance_id after each saved sample (remind user to lower & re-raise hand)",
    )
    args = p.parse_args(argv)

    try:
        target_labels = parse_labels_spec(args.label, args.labels)
    except ValueError as exc:
        p.error(str(exc))

    detector = None
    if args.roi_mode == "mediapipe":
        if not args.hand_model.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy MediaPipe hand_landmarker.task tại {args.hand_model}. "
                "Hãy dùng --roi-mode manual hoặc truyền đúng --hand-model."
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.hand_model)),
            num_hands=2,
            min_hand_detection_confidence=0.4,
        )
        detector = vision.HandLandmarker.create_from_options(opts)
    else:
        mp = None  # type: ignore[assignment]

    try:
        cap, backend_used = open_camera_with_fallback(args.camera_index)
    except RuntimeError:
        if detector is not None:
            detector.close()
        raise
    print(f"[CAPTURE] Đã mở camera index={args.camera_index} bằng backend={backend_used}")
    if len(target_labels) > 1:
        print(
            f"[CAPTURE] Chế độ thu tuần tự {len(target_labels)} lớp ({', '.join(target_labels)}), "
            f"mỗi lớp {args.count} ảnh (tổng {len(target_labels) * args.count} ảnh)."
        )

    quit_requested = False
    total_saved_session = 0
    try:
        for label_pos, active_label in enumerate(target_labels, start=1):
            if quit_requested:
                break
            saved_count = 0
            next_info = suggest_next_gesture_instance_id(
                args.output, args.participant_id, args.session_id, active_label
            )
            use_auto_id = args.gesture_instance_id.strip().lower() == "auto" or len(target_labels) > 1
            if use_auto_id:
                current_instance_id = str(next_info["next_gesture_instance_id"])
                print(
                    f"[CAPTURE {label_pos}/{len(target_labels)}] Lớp '{active_label}' -> tự động chọn "
                    f"gesture_instance_id = {current_instance_id} "
                    f"(đã có {next_info['existing_instance_count']} lần giơ)"
                )
            else:
                current_instance_id = args.gesture_instance_id.strip()
                if current_instance_id in set(next_info["existing_instance_ids"]):  # type: ignore[arg-type]
                    print(
                        f"[CẢNH BÁO TRÙNG MÃ] '{current_instance_id}' đã tồn tại trong {args.output} "
                        f"cho ({args.participant_id}/{args.session_id}/{active_label}). "
                        f"Gợi ý mã kế tiếp chưa dùng: {next_info['next_gesture_instance_id']} (bấm phím N để tăng mã)."
                    )
            print(
                f"[CAPTURE] Đang thu lớp '{active_label}' ({label_pos}/{len(target_labels)}) | "
                f"participant={args.participant_id} | session={args.session_id} | instance={current_instance_id}"
            )
            print(
                "Phím điều khiển: SPACE = chụp & xem trước | N = tăng gesture_instance_id | "
                "K = bỏ qua sang lớp kế | ESC/Q = thoát"
            )
            while saved_count < args.count:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Không đọc được khung hình từ webcam.")

                display = frame.copy()
                crop_box = None
                source_dims = {"width": int(frame.shape[1]), "height": int(frame.shape[0])}
                border_touches: list[str] = []
                if detector is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    if res.hand_landmarks:
                        sel_hand, _ = select_largest_hand_landmarks(
                            res.hand_landmarks, frame.shape[1], frame.shape[0]
                        )
                        crop_box = compute_padded_hand_crop_box(
                            sel_hand, frame.shape[1], frame.shape[0], padding_ratio=0.18
                        )
                        if crop_box is not None:
                            x, y, w, h = crop_box["x"], crop_box["y"], crop_box["width"], crop_box["height"]
                            border_touches = detect_crop_border_touches(crop_box, source_dims)
                            box_color = (0, 200, 255) if border_touches else (16, 185, 129)
                            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)

                hud = (
                    f"[{label_pos}/{len(target_labels)}] Label:{active_label} | "
                    f"{args.participant_id}/{args.session_id}/{current_instance_id} | "
                    f"Saved:{saved_count}/{args.count} | SPACE:Cap N:Inst K:NextLbl ESC:Quit"
                )
                cv2.putText(display, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 220, 120), 2)
                if border_touches:
                    warn_txt = f"BORDER TOUCH ({','.join(border_touches)}): lui tay vao giua khung"
                    cv2.putText(display, warn_txt, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 200, 255), 2)
                cv2.imshow("SignVision Capture (Raw Unmirrored)", display)
                key = cv2.waitKey(30) & 255
                if key in (27, ord("q"), ord("Q")):
                    quit_requested = True
                    break
                if key in (ord("k"), ord("K")):
                    print(f"[CAPTURE] Bỏ qua phần còn lại của lớp '{active_label}', chuyển lớp tiếp theo.")
                    break
                if key in (ord("n"), ord("N")):
                    current_instance_id = increment_instance_id(current_instance_id)
                    print(f"[CAPTURE] Đã chuyển sang gesture_instance_id = {current_instance_id}")
                    continue
                if key == 32:  # SPACE
                    if args.roi_mode == "mediapipe":
                        if crop_box is None:
                            print("[CẢNH BÁO] Chưa phát hiện bàn tay bằng MediaPipe — không lưu khung hình này.")
                            continue
                        x, y, w, h = crop_box["x"], crop_box["y"], crop_box["width"], crop_box["height"]
                        crop = frame[y : y + h, x : x + w]
                    else:
                        x, y, w, h = cv2.selectROI("Select hand; ENTER confirms ROI", frame)
                        if not w or not h:
                            continue
                        crop_box = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                        crop = frame[y : y + h, x : x + w]

                    cv2.imshow("Review crop: S=Save, other key=Discard", crop)
                    if (cv2.waitKey(0) & 255) in (ord("s"), ord("S")):
                        meta = {
                            "participant_id": args.participant_id,
                            "session_id": args.session_id,
                            "gesture_instance_id": current_instance_id,
                            "handedness": args.handedness,
                            "background": args.background,
                            "lighting": args.lighting,
                            "intended_sign": active_label,
                        }
                        folder = save_sample(
                            args.output,
                            crop,
                            meta,
                            roi_mode=args.roi_mode,
                            crop_box=crop_box,
                            source_frame_dimensions=source_dims,
                        )
                        saved_count += 1
                        total_saved_session += 1
                        saved_meta = json.loads((folder / "diagnostic_metadata.json").read_text(encoding="utf-8"))
                        dups = saved_meta.get("near_duplicate_warnings_at_capture", [])
                        touches = saved_meta.get("touches_frame_border", [])
                        print(f"[SAVED {active_label} {saved_count}/{args.count} | Tổng phiên: {total_saved_session}] {folder}")
                        if touches:
                            print(
                                f"  [CẢNH BÁO CHẠM MÉP] Khung tay chạm mép {touches} — lần sau nên lùi tay nhẹ vào giữa khung."
                            )
                        if dups:
                            print(
                                f"  [CẢNH BÁO GẦN TRÙNG] Phát hiện {len(dups)} ảnh tương tự (KHÔNG tự xóa): {dups[:3]}"
                            )
                        if args.auto_next_instance and saved_count < args.count:
                            current_instance_id = increment_instance_id(current_instance_id)
                            print(
                                f"  -> Hãy hạ tay xuống rồi giơ lại! gesture_instance_id tiếp theo: {current_instance_id}"
                            )
                    cv2.destroyWindow("Review crop: S=Save, other key=Discard")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
    print(f"[CAPTURE] Kết thúc phiên thu mẫu: đã lưu tổng {total_saved_session} ảnh vào {args.output} (label_confirmed=False).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
