import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import base64
import copy
import csv
import tempfile
from unittest.mock import patch
import cv2
import numpy as np
from src.model.diagnostic_storage import save_snapshot, FIELDS
from src.preprocess.build_hog_dataset import (
    apply_webcam_method_3b_from_bgr,
    extract_hog_with_diagnostics_from_bgr,
)
from manage_webcam_samples import load_saved_samples, export_collection_audit
from split_webcam_collection import validate


def url(image):
    return 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', image)[1]).decode()


crop = np.random.default_rng(7).integers(0, 256, (100, 80, 3), dtype=np.uint8)
m3b_crop, _, _ = apply_webcam_method_3b_from_bgr(crop, already_flipped=False)
legacy_gray = extract_hog_with_diagnostics_from_bgr(crop)[2]
m3b_gray = extract_hog_with_diagnostics_from_bgr(m3b_crop)[2]

p = dict.fromkeys(FIELDS, 'test')
p.update(
    handedness='right',
    frame_id='F_TEST',
    intended_sign='A',
    flip_input_for_model=False,
    preview_mirrored=True,
    experimental_webcam_mode=False,
    webcam_mode='legacy',
    received_crop_png_data_url=url(crop),
    preprocessed_gray_64x64_png_data_url=url(legacy_gray),
    same_frame_comparison={
        'legacy': {'frame_id': 'F_TEST', 'preprocessed_gray_64x64_png_data_url': url(legacy_gray)},
        'method_3b': {'frame_id': 'F_TEST', 'preprocessed_gray_64x64_png_data_url': url(m3b_gray)},
    },
)

with tempfile.TemporaryDirectory(prefix='signvision_test_') as t:
    root = Path(t) / 'samples'
    for kind in ('metadata', 'image', 'frame', 'gray_mismatch'):
        bad = copy.deepcopy(p)
        if kind == 'metadata':
            bad.pop('session_id')
        if kind == 'image':
            bad['received_crop_png_data_url'] = 'broken'
        if kind == 'frame':
            bad['same_frame_comparison']['legacy']['frame_id'] = 'wrong'
        if kind == 'gray_mismatch':
            bad['same_frame_comparison']['legacy']['preprocessed_gray_64x64_png_data_url'] = url(np.zeros((64, 64), np.uint8))
        try:
            save_snapshot(bad, root)
        except ValueError:
            pass
        else:
            raise AssertionError(kind)
        assert not root.exists()
        print('PASS reject', kind)

    with patch.object(Path, 'write_bytes', side_effect=OSError('simulated')):
        try:
            save_snapshot(p, root)
        except OSError:
            pass
        else:
            raise AssertionError('write failure')
    assert not list(root.iterdir())
    print('PASS write failure cleanup')

    result = save_snapshot(p, root)
    samples = load_saved_samples(root)
    assert len(samples) == 1 and not samples[0]['label_confirmed']
    assert samples[0]['raw_meta']['frame_id'] == 'F_TEST'
    assert samples[0]['raw_meta'].get('saved_at')
    export_collection_audit(samples, Path(t) / 'audit')
    with (Path(t) / 'audit/collection_metadata.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    errors = validate(rows, {'test': 'train'})[0]
    assert any('not confirmed' in e for e in errors)
    assert any('missing class' in e for e in errors)
    print('PASS save -> review reader -> audit -> split refusal')
