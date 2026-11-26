import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import {
  config,
  LANDMARK_SCOPES,
  type LandmarkScope,
  type Prediction,
  type TopKItem,
} from "./config";
import {
  advanceGate,
  applyMajorityVote,
  prepareLandmarkPayload,
} from "./predictionLogic";
import type { HandFrame } from "./useHandLandmarks";

export type ScopeHealthInfo = {
  available: boolean;
  label: string;
  classCount?: number;
  labels: string[];
  error?: string;
};

export type BackendHealth = {
  status: "checking" | "ready" | "offline";
  landmarkAvailable: boolean;
  featureSchema?: string;
  scopes: Record<LandmarkScope, ScopeHealthInfo>;
  error?: string;
};

const EMPTY_SCOPES: Record<LandmarkScope, ScopeHealthInfo> = {
  all_36: { available: false, label: "Tất cả 36", labels: [] },
  letters_az: { available: false, label: "Chữ cái A–Z", labels: [] },
  digits_09: { available: false, label: "Chữ số 0–9", labels: [] },
};

export function useModelLabels() {
  const [labels, setLabels] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const abort = new AbortController();
    setError("");
    void fetch(config.labelsUrl, { signal: abort.signal })
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then((data: unknown) => {
        const values = Array.isArray(data)
          ? data
          : data && typeof data === "object"
            ? Object.keys(data)
                .sort((a, b) => +a - +b)
                .map((k) => (data as Record<string, unknown>)[k])
            : [];
        if (
          !values.length ||
          values.some((v) => typeof v !== "string" || !v.trim()) ||
          new Set(values).size !== values.length
        )
          throw Error();
        setLabels(values as string[]);
      })
      .catch(() => {
        if (!abort.signal.aborted)
          setError(
            "Chưa tải được labels.json. Kiểm tra public/model/labels.json hoặc khởi động máy chủ SVM API.",
          );
      });
    return () => abort.abort();
  }, [attempt]);

  return { labels, error, retry: () => setAttempt((v) => v + 1) };
}

export function usePredictor(
  video: RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  scope: LandmarkScope | null,
  requestsPerSecond: number,
  smoothingWindow: number,
  stability: number,
  onAccept: (p: Prediction) => void,
) {
  const { labels: staticLabels, error: labelsError, retry: retryLabels } = useModelLabels();
  const [apiLabels, setApiLabels] = useState<string[]>([]);
  const labels = apiLabels.length ? apiLabels : staticLabels;

  const [attempt, setAttempt] = useState(0);
  const [backend, setBackend] = useState<BackendHealth>({
    status: "checking",
    landmarkAvailable: false,
    scopes: EMPTY_SCOPES,
  });
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [error, setError] = useState("");

  const state = useRef({
    busy: false,
    generation: 0,
    frameSeq: 0,
    last: -Infinity,
    label: "",
    count: 0,
    accepted: "",
    recentLabels: [] as string[],
  });
  const abort = useRef<AbortController | null>(null);
  const accept = useRef(onAccept);
  accept.current = onAccept;

  const reset = useCallback((clearAccepted = false) => {
    const s = state.current;
    s.generation++;
    s.busy = false;
    s.label = "";
    s.count = 0;
    s.recentLabels = [];
    if (clearAccepted) {
      s.accepted = "";
    }
    abort.current?.abort();
    setPrediction(null);
  }, []);

  // Immediately clear smoothing buffer & accepted state whenever scope changes
  useEffect(() => {
    reset(true);
  }, [scope, reset]);

  useEffect(() => {
    reset(false);
  }, [enabled, requestsPerSecond, smoothingWindow, stability, reset]);

  useEffect(
    () => () => {
      reset(false);
    },
    [reset],
  );

  // Check Python Landmark SVM backend readiness (GET /health)
  useEffect(() => {
    const controller = new AbortController();
    setBackend({
      status: "checking",
      landmarkAvailable: false,
      scopes: EMPTY_SCOPES,
    });
    const timer = setTimeout(() => controller.abort(), 3000);

    void fetch(config.healthUrl, { signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return (await r.json()) as {
          labels?: string[];
          landmark_v2_available?: boolean;
          landmark_v2?: {
            available?: boolean;
            feature_schema?: string | null;
            error?: string | null;
            scopes?: Record<
              string,
              {
                available?: boolean;
                label?: string;
                class_count?: number | null;
                labels?: string[];
                error?: string | null;
              }
            >;
          };
        };
      })
      .then((info) => {
        if (Array.isArray(info.labels) && info.labels.length > 0) {
          setApiLabels(info.labels);
        }
        const rawScopes = info.landmark_v2?.scopes || {};
        const parsedScopes: Record<LandmarkScope, ScopeHealthInfo> = {
          all_36: {
            available: Boolean(rawScopes.all_36?.available ?? info.landmark_v2_available),
            label: rawScopes.all_36?.label || "Tất cả 36",
            classCount: rawScopes.all_36?.class_count ?? undefined,
            labels: Array.isArray(rawScopes.all_36?.labels) ? rawScopes.all_36.labels : [],
            error: rawScopes.all_36?.error ?? undefined,
          },
          letters_az: {
            available: Boolean(rawScopes.letters_az?.available),
            label: rawScopes.letters_az?.label || "Chữ cái A–Z",
            classCount: rawScopes.letters_az?.class_count ?? undefined,
            labels: Array.isArray(rawScopes.letters_az?.labels) ? rawScopes.letters_az.labels : [],
            error: rawScopes.letters_az?.error ?? undefined,
          },
          digits_09: {
            available: Boolean(rawScopes.digits_09?.available),
            label: rawScopes.digits_09?.label || "Chữ số 0–9",
            classCount: rawScopes.digits_09?.class_count ?? undefined,
            labels: Array.isArray(rawScopes.digits_09?.labels) ? rawScopes.digits_09.labels : [],
            error: rawScopes.digits_09?.error ?? undefined,
          },
        };
        const anyAvail =
          parsedScopes.all_36.available ||
          parsedScopes.letters_az.available ||
          parsedScopes.digits_09.available;

        setBackend({
          status: "ready",
          landmarkAvailable: anyAvail,
          featureSchema: info.landmark_v2?.feature_schema ?? undefined,
          scopes: parsedScopes,
          error: anyAvail
            ? undefined
            : info.landmark_v2?.error ||
              "Máy chủ đã kết nối nhưng chưa nạp được model Landmark SVM nào.",
        });
        setError("");
      })
      .catch(() => {
        const msg = !controller.signal.aborted
          ? `Không kết nối được máy chủ nhận diện tại ${config.landmarkApiUrl}. Hãy khởi động backend Python rồi nhấn Thử lại.`
          : `Máy chủ nhận diện (${config.healthUrl}) không phản hồi sau 3 giây.`;
        setBackend({
          status: "offline",
          landmarkAvailable: false,
          scopes: EMPTY_SCOPES,
          error: msg,
        });
      })
      .finally(() => clearTimeout(timer));

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [attempt]);

  const predict = useCallback(
    async (frame: HandFrame) => {
      if (!enabled || error || backend.status !== "ready" || !scope) return;
      if (!backend.scopes[scope]?.available) return;

      if (!frame.hands.length) {
        reset(false);
        return;
      }

      const s = state.current;
      const minIntervalMs = 1000 / Math.max(1, requestsPerSecond);
      if (s.busy || frame.timestamp - s.last < minIntervalMs) {
        return;
      }

      const el = video.current;
      if (!el || el.videoWidth <= 0 || el.videoHeight <= 0) return;

      const activeHand =
        (frame.selectedHandIndex >= 0 ? frame.hands[frame.selectedHandIndex] : undefined) ??
        frame.hands[0];
      const landmarkPayload = prepareLandmarkPayload(activeHand, el.videoWidth, el.videoHeight);
      if (!landmarkPayload) return;

      s.busy = true;
      s.last = frame.timestamp;
      const generation = s.generation;
      const frameId = `F${++s.frameSeq}_${Math.round(frame.timestamp)}`;
      const started = performance.now();
      const controller = new AbortController();
      abort.current = controller;
      const timeout = setTimeout(() => controller.abort(), 3000);

      try {
        const response = await fetch(config.landmarkApiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...landmarkPayload,
            scope,
            top_k: 3,
            frame_id: frameId,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          let serverMsg = `Máy chủ Landmark SVM trả về HTTP ${response.status}`;
          try {
            const errJson = (await response.json()) as { error?: string };
            if (errJson?.error) serverMsg = errJson.error;
          } catch {
            /* ignore non-JSON body */
          }
          throw new Error(serverMsg);
        }

        const raw = (await response.json()) as {
          frame_id?: string;
          scope?: LandmarkScope;
          scope_label?: string;
          label?: string;
          predicted_label?: string;
          decision_score?: number;
          predicted_decision_score?: number;
          margin_top1_minus_top2?: number;
          confidence?: number;
          top_k?: TopKItem[];
          top_k_predictions?: TopKItem[];
          timing?: {
            feature_extract_ms?: number;
            svm_predict_ms: number;
            total_end_to_end_ms: number;
          };
        };

        if (generation !== s.generation) return;
        if (raw.frame_id && String(raw.frame_id) !== frameId) return;
        if (raw.scope && raw.scope !== scope) return;

        const allowedLabels =
          backend.scopes[scope]?.labels.length > 0 ? backend.scopes[scope].labels : labels;
        const rawLabel = String(raw.predicted_label ?? raw.label ?? "");
        const rawDecisionScore = Number(raw.predicted_decision_score ?? raw.decision_score ?? NaN);
        const margin = Number(raw.margin_top1_minus_top2 ?? 0);
        const confidence = Number(raw.confidence ?? NaN);
        const rawTopK = raw.top_k_predictions ?? raw.top_k ?? [];

        const validItem = (item: TopKItem) =>
          item &&
          typeof item.label === "string" &&
          (!allowedLabels.length || allowedLabels.includes(item.label)) &&
          Number.isFinite(Number(item.decision_score));

        if (
          !rawLabel ||
          (allowedLabels.length > 0 && !allowedLabels.includes(rawLabel)) ||
          !Number.isFinite(rawDecisionScore) ||
          !Number.isFinite(confidence) ||
          !Array.isArray(rawTopK) ||
          rawTopK.length === 0 ||
          !rawTopK.every(validItem)
        ) {
          throw new Error("Phản hồi từ máy chủ Landmark SVM không hợp lệ hoặc nhãn không khớp phạm vi.");
        }

        const winSize = Math.max(1, smoothingWindow);
        s.recentLabels = [...s.recentLabels, rawLabel].slice(-winSize);
        const voted = applyMajorityVote(s.recentLabels, winSize);
        const displayLabel = voted.label || rawLabel;

        const matchedTop = rawTopK.find((t) => t.label === displayLabel);
        const displayScore = matchedTop ? Number(matchedTop.decision_score) : rawDecisionScore;
        const scopeMeta = LANDMARK_SCOPES.find((item) => item.id === scope);

        const nextPrediction: Prediction = {
          frameId,
          scope,
          scopeLabel: raw.scope_label || scopeMeta?.label || scope,
          label: displayLabel,
          rawLabel,
          decisionScore: displayScore,
          rawDecisionScore,
          margin: Number.isFinite(margin) ? margin : 0,
          confidence,
          top_k: rawTopK.slice(0, 3).map((item, idx) => ({
            rank: item.rank ?? idx + 1,
            class_index: item.class_index,
            label: item.label,
            decision_score: Number(item.decision_score),
            ovr_votes: typeof item.ovr_votes === "number" ? item.ovr_votes : undefined,
            relative_weight:
              typeof item.relative_weight === "number" ? item.relative_weight : undefined,
          })),
          smoothed: winSize > 1,
          smoothingCount: voted.count,
          smoothingWindow: voted.total,
          timing: raw.timing,
        };

        setLatency(Math.round(performance.now() - started));
        setPrediction(nextPrediction);

        if (
          advanceGate(
            s,
            { label: displayLabel, decisionScore: displayScore, confidence },
            0,
            stability,
          )
        ) {
          accept.current(nextPrediction);
        }
      } catch (e) {
        if (generation === s.generation) {
          const msg = controller.signal.aborted
            ? "Máy chủ nhận diện không phản hồi sau 3 giây. Kiểm tra backend Python và thử lại."
            : e instanceof Error
              ? e.message
              : "Không thể kết nối máy chủ Landmark SVM.";
          setError(msg);
          setBackend((prev) => ({ ...prev, status: "offline", error: msg }));
          reset(false);
        }
      } finally {
        clearTimeout(timeout);
        if (generation === s.generation) s.busy = false;
      }
    },
    [enabled, error, backend, scope, labels, requestsPerSecond, smoothingWindow, stability, video, reset],
  );

  return {
    prediction,
    latency,
    backend,
    labels,
    error: error || backend.error || (!apiLabels.length ? labelsError : ""),
    loading: backend.status === "checking",
    predict,
    reset: () => reset(false),
    retry: () => {
      reset(true);
      setError("");
      setAttempt((v) => v + 1);
      if (labelsError) retryLabels();
    },
  };
}
