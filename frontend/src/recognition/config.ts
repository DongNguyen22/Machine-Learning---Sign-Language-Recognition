const env = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env;
const numeric = (key: string, fallback: number, min: number, max: number) => {
  const raw = env[key];
  if (raw === undefined || raw === "") return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
};

export type LandmarkScope = "all_36" | "letters_az" | "digits_09";

export const LANDMARK_SCOPES: {
  id: LandmarkScope;
  label: string;
  badge: string;
  description: string;
}[] = [
  {
    id: "all_36",
    label: "Tất cả 36",
    badge: "LANDMARK SVM · 36",
    description: "Nhận diện cả 36 ký hiệu (0–9, A–Z)",
  },
  {
    id: "letters_az",
    label: "Chữ cái A–Z",
    badge: "LANDMARK SVM · A–Z",
    description: "Mô hình chuyên biệt 26 chữ cái A–Z (tránh nhầm sang chữ số)",
  },
  {
    id: "digits_09",
    label: "Chữ số 0–9",
    badge: "LANDMARK SVM · 0–9",
    description: "Mô hình chuyên biệt 10 chữ số 0–9",
  },
];

export type TopKItem = {
  rank?: number;
  class_index?: number;
  label: string;
  decision_score: number;
  ovr_votes?: number;
  relative_weight?: number;
};

export type PredictionTiming = {
  feature_extract_ms?: number;
  svm_predict_ms: number;
  total_end_to_end_ms: number;
};

export type Prediction = {
  frameId: string;
  scope: LandmarkScope;
  scopeLabel: string;
  label: string;
  rawLabel: string;
  decisionScore: number;
  rawDecisionScore: number;
  margin: number;
  confidence: number;
  top_k: TopKItem[];
  smoothed: boolean;
  smoothingCount: number;
  smoothingWindow: number;
  timing?: PredictionTiming;
};

export type HistoryEntry = {
  label: string;
  decisionScore: number;
  confidence: number;
  margin: number;
  scope: LandmarkScope;
  time: string;
  id: number;
};

const defaultApiUrl = env.VITE_API_URL || "http://localhost:8000/predict";
const baseApiOrigin = defaultApiUrl.endsWith("/predict")
  ? defaultApiUrl.slice(0, -"/predict".length)
  : "http://localhost:8000";

export const config = {
  landmarkApiUrl: env.VITE_LANDMARK_API_URL || `${baseApiOrigin}/predict/landmark`,
  healthUrl: env.VITE_API_HEALTH_URL || `${baseApiOrigin}/health`,
  labelsUrl: env.VITE_LABELS_URL || "/model/labels.json",
  stability: Math.round(numeric("VITE_STABILITY", 4, 1, 60)),
  smoothingWindow: Math.round(numeric("VITE_SMOOTHING_WINDOW", 3, 1, 9)),
  requestsPerSecond: numeric("VITE_PREDICTION_HZ", 4, 1, 10),
  wasmUrl: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm",
  handModelUrl:
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
};
