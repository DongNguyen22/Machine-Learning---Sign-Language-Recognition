"""Validate complete snapshots before publishing an atomic sample directory."""
import base64
from datetime import datetime
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
import cv2
import numpy as np

FIELDS = ('participant_id', 'session_id', 'gesture_instance_id', 'handedness', 'background', 'lighting')
NEAR_DUPLICATE_MAE_THRESHOLD = 8.0
BORDER_TOUCH_MARGIN_PX = 2


def detect_crop_border_touches(
    crop_box: dict | None,
    source_frame_dimensions: dict | None,
    margin_px: int = BORDER_TOUCH_MARGIN_PX,
) -> list[str]:
    """Return touched camera frame edges ('left', 'top', 'right', 'bottom') without altering crop logic."""
    if not isinstance(crop_box, dict) or not isinstance(source_frame_dimensions, dict):
        return []
    try:
        x = float(crop_box['x'])
        y = float(crop_box['y'])
        w = float(crop_box['width'])
        h = float(crop_box['height'])
        fw = float(source_frame_dimensions.get('width') or source_frame_dimensions.get('frame_width') or 0)
        fh = float(source_frame_dimensions.get('height') or source_frame_dimensions.get('frame_height') or 0)
    except (KeyError, TypeError, ValueError):
        return []
    if fw <= 1 or fh <= 1 or w <= 0 or h <= 0:
        return []
    margin = max(0.0, float(margin_px))
    touched: list[str] = []
    if x <= margin:
        touched.append('left')
    if y <= margin:
        touched.append('top')
    if (x + w) >= (fw - margin):
        touched.append('right')
    if (y + h) >= (fh - margin):
        touched.append('bottom')
    return touched


def suggest_next_gesture_instance_id(
    root: Path | str,
    participant_id: str,
    session_id: str,
    label: str,
) -> dict[str, object]:
    """Scan existing samples for (participant_id, session_id, label) and suggest the next unused <LABEL>_G0N."""
    pid = str(participant_id or '').strip()
    sid = str(session_id or '').strip()
    lbl = str(label or '').strip().upper()
    existing_ids: set[str] = set()
    used_nums: set[int] = set()
    root_p = Path(root)
    if root_p.is_dir() and pid and lbl:
        pattern = re.compile(rf'^{re.escape(lbl)}_G(\d+)$', re.IGNORECASE)
        fallback_num = re.compile(r'(\d+)$')
        for folder in root_p.iterdir():
            if not folder.is_dir() or folder.name.startswith('.pending_'):
                continue
            meta_p = folder / 'diagnostic_metadata.json'
            if not meta_p.is_file():
                continue
            try:
                m = json.loads(meta_p.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            m_pid = str(m.get('participant_id', '') or '').strip()
            m_sid = str(m.get('session_id', '') or '').strip()
            m_lbl = str(m.get('verified_true_label') or m.get('intended_sign') or '').strip().upper()
            m_gid = str(m.get('gesture_instance_id', '') or '').strip()
            if m_pid != pid or m_lbl != lbl or not m_gid:
                continue
            if sid and m_sid and m_sid != sid:
                continue
            existing_ids.add(m_gid)
            match = pattern.match(m_gid) or fallback_num.search(m_gid)
            if match:
                used_nums.add(int(match.group(1)))

    next_num = (max(used_nums) + 1) if used_nums else 1
    suggested = f'{lbl}_G{next_num:02d}' if lbl else f'G{next_num:02d}'
    while suggested in existing_ids:
        next_num += 1
        suggested = f'{lbl}_G{next_num:02d}' if lbl else f'G{next_num:02d}'
    return {
        'participant_id': pid,
        'session_id': sid,
        'label': lbl,
        'existing_instance_ids': sorted(existing_ids),
        'existing_instance_count': len(existing_ids),
        'next_gesture_instance_id': suggested,
    }


def _find_near_duplicates(root: Path, crop_png_bytes: bytes, legacy_gray_64: np.ndarray) -> list[dict[str, object]]:
    """Scan existing sample folders for exact/near duplicates (warning only; never deletes or blocks)."""
    warnings: list[dict[str, object]] = []
    if not root.is_dir():
        return warnings
    cand_sha = hashlib.sha256(crop_png_bytes).hexdigest()
    cand_small = cv2.resize(legacy_gray_64, (32, 32)).astype(np.float32)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.pending_') or not (folder / 'diagnostic_metadata.json').is_file():
            continue
        exact_dup = False
        crop_p = folder / 'received_crop.png'
        if crop_p.is_file():
            try:
                if hashlib.sha256(crop_p.read_bytes()).hexdigest() == cand_sha:
                    exact_dup = True
            except OSError:
                pass
        mae = None
        gray_p = folder / 'legacy_gray_64x64.png'
        if not gray_p.is_file():
            gray_p = folder / 'preprocessed_gray_64x64.png'
        if gray_p.is_file():
            try:
                raw = gray_p.read_bytes()
                prev = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if prev is not None:
                    prev_small = cv2.resize(prev, (32, 32)).astype(np.float32)
                    mae = float(np.mean(np.abs(prev_small - cand_small)))
            except OSError:
                pass
        if exact_dup or (mae is not None and mae <= NEAR_DUPLICATE_MAE_THRESHOLD):
            warnings.append({
                'existing_folder': folder.name,
                'exact_sha256_match': exact_dup,
                'gray_mae': round(mae, 4) if mae is not None else None,
            })
    return warnings


def save_snapshot(payload, root):
    for key in (*FIELDS, 'frame_id', 'intended_sign'):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError('Missing or invalid field: ' + key)
    if payload['handedness'] not in ('left', 'right') or payload['intended_sign'] not in list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        raise ValueError('Invalid handedness or intended_sign')
    for key in ('flip_input_for_model', 'experimental_webcam_mode', 'preview_mirrored'):
        if type(payload.get(key)) is not bool:
            raise ValueError('Expected boolean: ' + key)
    mode = 'method_3b' if payload['experimental_webcam_mode'] else 'legacy'
    if payload.get('webcam_mode') != mode:
        raise ValueError('webcam_mode disagrees with experimental_webcam_mode')
    comparison = payload.get('same_frame_comparison', {})
    images = {}
    def decode(url, name, gray=False):
        try:
            raw = base64.b64decode(url.split(',', 1)[1], validate=True)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None or (gray and image.shape != (64, 64)):
                raise ValueError()
        except Exception as exc:
            raise ValueError('Unreadable image or invalid dimensions: ' + name) from exc
        images[name] = raw
        return image
    crop = decode(payload.get('received_crop_png_data_url'), 'received_crop.png')
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError('Crop must be a BGR color image')
    from src.preprocess.build_hog_dataset import apply_webcam_method_3b_from_bgr, extract_hog_with_diagnostics_from_bgr
    # /predict diagnostics export the canonical UNFLIPPED crop.
    processed, _, _ = apply_webcam_method_3b_from_bgr(crop, already_flipped=False)
    computed = {k: extract_hog_with_diagnostics_from_bgr(v)[2] for k, v in
                [('legacy', crop), ('method_3b', processed)]}
    active = decode(payload.get('preprocessed_gray_64x64_png_data_url'), 'preprocessed_gray_64x64.png', True)
    for method in ('legacy', 'method_3b'):
        entry = comparison.get(method, {})
        if entry.get('frame_id') != payload['frame_id']:
            raise ValueError('frame_id mismatch: ' + method)
        gray = decode(entry.get('preprocessed_gray_64x64_png_data_url'), method + '_gray_64x64.png', True)
        if not np.array_equal(gray, computed[method]):
            raise ValueError('Client image differs from backend reconstruction: ' + method)
        images[method + '_gray_64x64.png'] = cv2.imencode('.png', computed[method])[1].tobytes()
    active_expected = computed[mode]
    if mode == 'legacy' and payload['flip_input_for_model']:
        active_expected = extract_hog_with_diagnostics_from_bgr(cv2.flip(crop, 1))[2]
    if not np.array_equal(active, active_expected):
        raise ValueError('Active HOG image differs from backend reconstruction')
    images['preprocessed_gray_64x64.png'] = cv2.imencode('.png', active_expected)[1].tobytes()
    root = Path(root)
    near_dups = _find_near_duplicates(root, images['received_crop.png'], computed['legacy'])
    touches_frame_border = detect_crop_border_touches(
        payload.get('crop_box'),
        payload.get('source_frame_dimensions'),
    )
    border_touch_warning = len(touches_frame_border) > 0
    root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    staging, final = root / ('.pending_' + token), root / ('sample_' + token)
    staging.mkdir()
    try:
        for name, raw in images.items():
            (staging / name).write_bytes(raw)
        meta = {k: v for k, v in payload.items() if not k.endswith('_data_url') and k != 'same_frame_comparison'}
        clean_comparison = {}
        for method in ('legacy', 'method_3b'):
            entry = comparison.get(method, {})
            if isinstance(entry, dict):
                clean_comparison[method] = {
                    k: v for k, v in entry.items() if not k.endswith('_data_url')
                }
        meta.update(
            saved_at=datetime.now().isoformat(),
            label_confirmed=False,
            image_frame_id=payload['frame_id'],
            crop_orientation='canonical_unflipped_from_predict',
            gray_source='backend_recomputed_from_saved_crop',
            method_3b_flip_count_from_saved_crop=1,
            same_frame_comparison=clean_comparison,
            near_duplicate_warnings_at_capture=near_dups,
            touches_frame_border=touches_frame_border,
            border_touch_warning=border_touch_warning,
        )
        (staging / 'diagnostic_metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging)
        raise
    next_info = suggest_next_gesture_instance_id(
        root,
        payload['participant_id'],
        payload['session_id'],
        payload['intended_sign'],
    )
    return {
        'status': 'saved',
        'frame_id': payload['frame_id'],
        'sample_directory': str(final.resolve()),
        'near_duplicate_warnings': near_dups,
        'touches_frame_border': touches_frame_border,
        'border_touch_warning': border_touch_warning,
        'next_suggested_gesture_instance_id': next_info['next_gesture_instance_id'],
        'existing_instance_count_for_label': next_info['existing_instance_count'],
    }
