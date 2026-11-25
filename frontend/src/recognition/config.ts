const env = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env;
const numeric = (key: string, fallback: number, min: number, max: number) =>
  Math.max(min, Math.min(max, Number(env[key]) || fallback));
export const config = {
  mode: env.VITE_MODEL_MODE === "browser" ? "browser" : "api",
  apiUrl: env.VITE_API_URL || "http://localhost:8000/predict",
  input: env.VITE_MODEL_INPUT === "image" ? "image" : "landmarks",
  modelUrl: env.VITE_MODEL_URL || "/model/model.json",
  labelsUrl: env.VITE_LABELS_URL || "/model/labels.json",
  threshold: numeric("VITE_CONFIDENCE_THRESHOLD", 0.7, 0, 1),
  windowSize: Math.round(numeric("VITE_WINDOW_SIZE", 30, 1, 120)),
  stability: Math.round(numeric("VITE_STABILITY", 5, 1, 60)),
  requestsPerSecond: numeric("VITE_PREDICTION_HZ", 7, 5, 10),
  imageSize: Math.round(numeric("VITE_IMAGE_SIZE", 224, 32, 640)),
  wasmUrl: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm",
  handModelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
export type Prediction = {
  label: string;
  confidence: number;
  top_k: { label: string; confidence: number }[];
};
export type HistoryEntry = { label: string; confidence: number; time: string; id: number };
