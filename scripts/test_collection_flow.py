import tempfile
import json
import csv
from pathlib import Path
import numpy as np
from capture_webcam_collection import save_sample
from manage_webcam_samples import main, load_saved_samples, export_collection_audit
from split_webcam_collection import validate

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    image = np.random.default_rng(42).integers(0, 256, (80, 90, 3), dtype=np.uint8)
    meta = dict(participant_id='P1', session_id='S1', gesture_instance_id='G1',
                handedness='right', background='plain', lighting='day', intended_sign='A')
    first = save_sample(root/'samples', image, meta)
    second = save_sample(root/'samples', image, dict(meta, participant_id='P2'))
    print('PASS save controlled images')
    edit = root/'edit.json'
    edit.write_text(json.dumps({first.name: {'lighting': 'lamp'}}))
    main(['list', '--diag-dir', str(root/'samples'), '--edit-metadata', str(edit)])
    saved = json.loads((first/'diagnostic_metadata.json').read_text())
    assert saved['lighting'] == 'lamp' and saved['label_confirmed'] is False
    print('PASS metadata review preserves label state')
    export_collection_audit(load_saved_samples(root/'samples'), root/'audit')
    with (root/'audit/collection_metadata.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    errors = validate(rows, {'P1': 'train', 'P2': 'test'})[0]
    assert any('Duplicate image' in e for e in errors)
    assert any('missing class' in e for e in errors)
    assert any('not confirmed' in e for e in errors)
    print('PASS audit and split reject duplicate identities, missing classes, unconfirmed labels')
