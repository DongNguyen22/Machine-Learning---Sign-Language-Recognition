import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { config, type Prediction } from "./config";
import { advanceGate } from "./predictionLogic";
import type { HandFrame } from "./useHandLandmarks";
export function useModelLabels() {
  const [labels, setLabels] = useState<string[]>([]),
    [error, setError] = useState("");
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
            "Chưa tải được labels.json. Cung cấp nhãn thật của model trong public/model/labels.json.",
          );
      });
    return () => abort.abort();
  }, [attempt]);
  return { labels, error, retry: () => setAttempt((v) => v + 1) };
}
export function usePredictor(
  video: RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  threshold: number,
  stability: number,
  onAccept: (p: Prediction) => void,
) {
  const { labels, error: labelsError, retry: retryLabels } = useModelLabels();
  const [attempt, setAttempt] = useState(0);
  const [prediction, setPrediction] = useState<Prediction | null>(null),
    [latency, setLatency] = useState<number | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  const state = useRef({
    busy: false,
    generation: 0,
    last: -Infinity,
    label: "",
    count: 0,
    accepted: "",
  });
  const abort = useRef<AbortController | null>(null);
  const model = useRef<import("@tensorflow/tfjs").LayersModel | null>(null);
  const accept = useRef(onAccept);
  accept.current = onAccept;
  const reset = useCallback(() => {
    const s = state.current;
    s.generation++;
    s.busy = false;
    s.label = "";
    s.count = 0;
    abort.current?.abort();
    setPrediction(null);
  }, []);
  useEffect(() => {
    reset();
  }, [enabled, threshold, stability, reset]);
  useEffect(
    () => () => {
      reset();
    },
    [reset],
  );
  useEffect(() => {
    if (config.mode !== "browser" || !enabled || model.current) {
      setLoading(false);
      return;
    }
    let disposed = false;
    setLoading(true);
    void import("@tensorflow/tfjs")
      .then(async (tf) => {
        await tf.ready();
        const loaded = await tf.loadLayersModel(config.modelUrl);
        if (disposed) loaded.dispose();
        else {
          model.current = loaded;
          setLoading(false);
        }
      })
      .catch(() => {
        if (!disposed) {
          setLoading(false);
          setError(
            "Không tải được TensorFlow.js model.json. Kiểm tra public/model/ và các tệp trọng số.",
          );
        }
      });
    return () => {
      disposed = true;
    };
  }, [enabled, attempt]);
  useEffect(
    () => () => {
      model.current?.dispose();
      model.current = null;
    },
    [],
  );
  const predict = useCallback(
    async (frame: HandFrame) => {
      if (!enabled || error) return;
      if (!frame.hands.length) {
        reset();
        return;
      }
      const s = state.current;
      if (
        s.busy ||
        !labels.length ||
        (config.input === "landmarks" && frame.sequence.length < config.windowSize) ||
        frame.timestamp - s.last < 1000 / config.requestsPerSecond
      )
        return;
      if (config.mode === "browser" && !model.current) return;
      s.busy = true;
      s.last = frame.timestamp;
      const generation = s.generation,
        started = performance.now();
      const controller = new AbortController();
      abort.current = controller;
      const timeout = setTimeout(() => controller.abort(), 3000);
      let timedOut = false;
      try {
        let raw: unknown;
        const el = video.current;
        if (config.mode === "api") {
          let body: { landmarks: number[][] } | { image: string } = { landmarks: frame.sequence };
          if (config.input === "image") {
            if (!el) return;
            const canvas = document.createElement("canvas");
            canvas.width = el.videoWidth;
            canvas.height = el.videoHeight;
            canvas.getContext("2d")!.drawImage(el, 0, 0);
            body = { image: canvas.toDataURL("image/jpeg", 0.8).split(",")[1] };
          }
          const response = await fetch(config.apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
          });
          if (!response.ok) throw Error(`Model trả về HTTP ${response.status}`);
          raw = await response.json();
        } else {
          const tf = await import("@tensorflow/tfjs");
          const output = tf.tidy(() => {
            const input =
              config.input === "image"
                ? tf.image
                    .resizeBilinear(tf.browser.fromPixels(el!), [
                      config.imageSize,
                      config.imageSize,
                    ])
                    .toFloat()
                    .div(255)
                    .expandDims(0)
                : tf.tensor3d([frame.sequence], [1, config.windowSize, 63]);
            const result = model.current!.predict(input);
            if (Array.isArray(result)) throw Error("Model cần một tensor đầu ra xác suất.");
            return result;
          });
          let scores: number[];
          try {
            scores = Array.from(await output.data());
          } finally {
            output.dispose();
          }
          if (scores.length !== labels.length)
            throw Error("Số lớp của model không khớp labels.json.");
          const ranked = scores
            .map((confidence, i) => ({ label: labels[i], confidence }))
            .sort((a, b) => b.confidence - a.confidence);
          raw = { ...ranked[0], top_k: ranked.slice(0, 3) };
          timedOut = performance.now() - started > 3000;
        }
        if (generation !== s.generation) return;
        if (timedOut) throw Error("Model chạy quá 3 giây.");
        const p = raw as Prediction;
        const valid = (v: { label: string; confidence: number }) =>
          v &&
          labels.includes(v.label) &&
          Number.isFinite(v.confidence) &&
          v.confidence >= 0 &&
          v.confidence <= 1;
        if (!valid(p) || !Array.isArray(p.top_k) || !p.top_k.every(valid))
          throw Error("Phản hồi model không hợp lệ hoặc nhãn không thuộc labels.json.");
        setLatency(Math.round(performance.now() - started));
        setPrediction(p);
        if (advanceGate(s, p, threshold, stability)) accept.current(p);
      } catch (e) {
        if (generation === s.generation) {
          setError(
            controller.signal.aborted
              ? "Model không phản hồi sau 3 giây. Kiểm tra máy chủ và thử lại."
              : e instanceof Error
                ? e.message
                : "Không thể kết nối model.",
          );
          reset();
        }
      } finally {
        clearTimeout(timeout);
        if (generation === s.generation) s.busy = false;
      }
    },
    [enabled, error, labels, threshold, stability, video, reset],
  );
  return {
    prediction,
    latency,
    error: error || labelsError,
    loading,
    predict,
    retry: () => {
      reset();
      setError("");
      setAttempt((v) => v + 1);
      if (labelsError) retryLabels();
    },
  };
}
