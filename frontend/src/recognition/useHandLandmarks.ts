import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { config } from "./config";
export type Point = { x: number; y: number; z: number };
export type HandFrame = { hands: Point[][]; sequence: number[][]; timestamp: number };
export function useHandLandmarks(
  video: RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  onFrame: (frame: HandFrame) => void,
) {
  const callback = useRef(onFrame);
  callback.current = onFrame;
  const [hands, setHands] = useState<Point[][]>([]);
  const [fps, setFps] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!enabled) {
      setHands([]);
      setFps(0);
      setLoading(false);
      return;
    }
    let disposed = false,
      raf = 0,
      lastTime = -1,
      samples = 0,
      start = performance.now();
    let detector: import("@mediapipe/tasks-vision").HandLandmarker | undefined;
    let sequence: number[][] = [];
    setLoading(true);
    setError("");
    async function init() {
      try {
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const files = await FilesetResolver.forVisionTasks(config.wasmUrl);
        const instance = await HandLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: config.handModelUrl },
          runningMode: "VIDEO",
          numHands: 2,
        });
        if (disposed) {
          instance.close();
          return;
        }
        detector = instance;
        setLoading(false);
        const loop = () => {
          if (disposed) return;
          const el = video.current;
          if (el && el.readyState >= 2 && el.currentTime !== lastTime) {
            lastTime = el.currentTime;
            const now = performance.now();
            try {
              const result = instance.detectForVideo(el, now);
              const current = result.landmarks;
              // The model contract is 63 values: use the first (most confident) hand, never mirrored.
              if (current.length) {
                sequence.push(current[0].flatMap((p) => [p.x, p.y, p.z]));
                sequence = sequence.slice(-config.windowSize);
              } else sequence = [];
              setHands(current);
              callback.current({ hands: current, sequence: [...sequence], timestamp: now });
              samples++;
              if (now - start >= 1000) {
                setFps(Math.round((samples * 1000) / (now - start)));
                samples = 0;
                start = now;
              }
            } catch (e) {
              setError(e instanceof Error ? e.message : "Không thể phát hiện bàn tay");
              return;
            }
          }
          raf = requestAnimationFrame(loop);
        };
        raf = requestAnimationFrame(loop);
      } catch {
        if (!disposed) {
          setLoading(false);
          setError("Không tải được MediaPipe. Kiểm tra kết nối mạng rồi thử lại.");
        }
      }
    }
    void init();
    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      detector?.close();
    };
  }, [enabled, video, retry]);
  return { hands, fps, loading, error, retry: () => setRetry((v) => v + 1) };
}
