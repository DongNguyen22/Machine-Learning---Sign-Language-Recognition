import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json, copy, threading, urllib.request, urllib.error, csv
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from http.server import ThreadingHTTPServer
from src.model.svm_api_server import create_request_handler
from src.model.predict_svm import SVMSignPredictor
from manage_webcam_samples import load_saved_samples, export_collection_audit, main
from split_webcam_collection import validate

root = Path('results/svm/http_tests') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
root.mkdir(parents=True)
predictor = SVMSignPredictor()
app = SimpleNamespace(allowed_origins=set(), max_payload_bytes=4000000, max_image_bytes=3000000, predictor=predictor, diagnostic_save_dir=root/'samples')
server = ThreadingHTTPServer(('127.0.0.1', 0), create_request_handler(app))
threading.Thread(target=server.serve_forever, daemon=True).start()
def post(p, endpoint='/diagnostic/save'):
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}{endpoint}', data=json.dumps(p).encode(), headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)
results = {}
try:
    image = predictor.encode_png_data_url(np.random.default_rng(42).integers(0,256,(100,80,3),dtype=np.uint8))
    status, prediction = post(dict(image=image, include_diagnostics=True, frame_id='HTTP_TEST', experimental_webcam_mode=True), '/predict')
    assert status == 200, prediction
    assert prediction['frame_id'] == 'HTTP_TEST'
    assert prediction['diagnostics']['frame_id'] == 'HTTP_TEST'
    frozen = copy.deepcopy(prediction)
    status, newer = post(dict(image=image, include_diagnostics=True, frame_id='NEWER_FRAME', experimental_webcam_mode=False), '/predict')
    assert status == 200 and newer['frame_id'] == 'NEWER_FRAME'
    assert frozen['frame_id'] == 'HTTP_TEST'
    prediction = frozen
    results['predict_http_and_frozen_payload'] = 'PASS HTTP 200; later response does not replace selected snapshot (simulated client)'
    diag = prediction['diagnostics']
    p = dict(participant_id='P_TEST', session_id='S_TEST', gesture_instance_id='G_TEST', handedness='right', background='synthetic', lighting='synthetic', intended_sign='A', frame_id='HTTP_TEST', flip_input_for_model=False, preview_mirrored=True, experimental_webcam_mode=True, webcam_mode='method_3b')
    for k in ('received_crop_png_data_url','preprocessed_gray_64x64_png_data_url'): p[k] = diag[k]
    p['same_frame_comparison'] = copy.deepcopy(prediction['same_frame_comparison'])
    for v in p['same_frame_comparison'].values(): v['frame_id'] = p['frame_id']
    for case in ('metadata','corrupt','frame','gray'):
        bad = copy.deepcopy(p)
        if case == 'metadata': bad.pop('session_id')
        if case == 'corrupt': bad['received_crop_png_data_url'] = 'broken'
        if case == 'frame': bad['same_frame_comparison']['legacy']['frame_id'] = 'wrong'
        if case == 'gray': bad['same_frame_comparison']['legacy']['preprocessed_gray_64x64_png_data_url'] = predictor.encode_png_data_url(np.zeros((64,64),np.uint8))
        status, body = post(bad)
        assert status == 400, (case,status,body)
        assert not app.diagnostic_save_dir.exists()
        results[case] = {'status':status,'result':'PASS','response':body}
    with patch.object(Path,'write_bytes',side_effect=OSError('controlled write failure')):
        status, body = post(p)
    assert status == 500 and not list(app.diagnostic_save_dir.iterdir())
    results['write_failure'] = 'PASS HTTP 500, no partial sample'
    status, body = post(p)
    assert status == 200, body
    sample = Path(body['sample_directory'])
    assert len(list(sample.glob('*.png'))) == 4
    meta = json.loads((sample/'diagnostic_metadata.json').read_text())
    assert meta['frame_id'] == 'HTTP_TEST' and meta['label_confirmed'] is False
    export_collection_audit(load_saved_samples(app.diagnostic_save_dir), root/'audit')
    with (root/'audit/collection_metadata.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    errors=validate(rows, {'P_TEST':'train'})[0]
    assert any('not confirmed' in e for e in errors) and any('missing class' in e for e in errors)
    results['save_review_audit_split'] = {'result':'PASS','sample':str(sample),'errors':errors}
finally:
    server.shutdown()
    server.server_close()
    (root/'results.json').write_text(json.dumps(results,indent=2))
print('RESULTS',root)
