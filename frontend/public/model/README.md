# SignVision model assets

No model weights or labels were supplied. The app never substitutes fake predictions.

Place the **real labels** at `public/model/labels.json`, either a nonempty array of
strings in class-index order, or an object mapping numeric indices to strings.
The dictionary uses this file too. API responses with unknown labels are rejected.

## Backend mode (default)

Supply a server at `http://localhost:8000/predict` or set `VITE_API_URL`.
The front end sends `{"landmarks":[...frames]}`: 30 rows, each containing 63
normalized values in MediaPipe landmark order `[x0,y0,z0,...,x20,y20,z20]`.
For multiple visible hands, only the first MediaPipe result is sent, matching
the requested single-hand input shape. Coordinates are not mirrored for inference.

Return probabilities from 0–1:
`{"label":"<real model label>","confidence":0.92,"top_k":[{"label":"<real model label>","confidence":0.92}]}`

Calls are serialized, limited to 7 requests/second, aborted on pause, missing hand,
settings changes or leaving the screen, and time out after 3 seconds.
The backend must allow CORS from this app's exact origin and allow JSON POST.
`localhost` refers to the user's device, not the hosted Make environment.
For a hosted HTTPS app, use an HTTPS API endpoint; browser mixed-content,
local-network permissions, and CORS may otherwise block the request.
This app does not implement or deploy the external prediction server.

## Browser mode

Set `VITE_MODEL_MODE=browser`. Place a TensorFlow.js **LayersModel**
`model.json` and its weight shards in this folder.
Landmark input must be `[1, VITE_WINDOW_SIZE, 63]`; output must be a single
`[1, numberOfLabels]` tensor of probabilities (not logits).
ONNX, GraphModel and WebSocket transports are not implemented.

## Image input

Set `VITE_MODEL_INPUT=image`.
API mode sends `{"image":"<raw base64 JPEG without data-URL prefix>"}` using
the unmirrored camera resolution. Browser mode uses `[1,size,size,3]` RGB
float32 normalized to 0–1, where size defaults to 224.
Adjust preprocessing if your trained model uses a different contract.
No hand means no inference in either mode.

## Configuration

Defaults are in `src/recognition/config.ts`. Optionally create `.env.local`:

```dotenv
VITE_MODEL_MODE=api
VITE_API_URL=http://localhost:8000/predict
VITE_MODEL_INPUT=landmarks
VITE_MODEL_URL=/model/model.json
VITE_LABELS_URL=/model/labels.json
VITE_CONFIDENCE_THRESHOLD=0.7
VITE_WINDOW_SIZE=30
VITE_STABILITY=5
VITE_PREDICTION_HZ=7
VITE_IMAGE_SIZE=224
```

VITE variables are public, so never put secrets in them.
Restart the host-managed preview after changing environment variables if needed.
MediaPipe WASM and the standard hand-landmarker task download from the URLs in
config.ts; vendor these assets and update those URLs for offline deployments.

## Checks

```sh
node --experimental-strip-types --test src/recognition/predictionLogic.test.mjs
pnpm exec tsc --noEmit
pnpm build
```

Real-device acceptance checks require a camera and the actual trained model:
grant/deny permissions, unplug/switch camera, use another app to occupy it,
show/remove a hand, pause/unpause, verify prediction stability and sentence
deduplication, stop the API to test timeout/retry, then leave the recognition
page and verify the camera indicator turns off.
