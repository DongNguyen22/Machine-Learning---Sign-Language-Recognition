import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { config } from "./config";
import { selectLargestHand } from "./predictionLogic";

export type Point = { x: number; y: number; z: number };
export type HandFrame = {
  hands: Point[][];
  selectedHandIndex: number;
  timestamp: number;
};

export function useHandLandmarks(
  video: RefObject<HTMLVideoElement | null>,
  enabled: boolean,
  onFrame: (frame: HandFrame) => void,
) {
  const callback = useRef(onFrame);
  callback.current = onFrame;
  const [hands, setHands] = useState<Point[][]>([]);
  const [selectedHandIndex, setSelectedHandIndex] = useState<number>(-1);
  const [fps, setFps] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (!enabled) {
      setHands([]);
      setSelectedHandIndex(-1);
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

    setError("");
    setLoading(true);

    async function initMediaPipe() {
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
          if (el && el.readyState >= 2 && el.videoWidth > 0 && el.currentTime !== lastTime) {
            lastTime = el.currentTime;
            const now = performance.now();
            try {
              const result = instance.detectForVideo(el, now);
              const current = result.landmarks;
              let chosenIdx = -1;

              if (current.length > 0) {
                const { selectedIndex } = selectLargestHand(
                  current,
                  el.videoWidth,
                  el.videoHeight,
                );
                chosenIdx = selectedIndex;
              }

              setHands(current);
              setSelectedHandIndex(chosenIdx);
              callback.current({
                hands: current,
                selectedHandIndex: chosenIdx,
                timestamp: now,
              });

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
          setError(
            "Không tải được bộ phát hiện bàn tay MediaPipe. Hãy kiểm tra kết nối mạng rồi nhấn Thử lại.",
          );
        }
      }
    }
    void initMediaPipe();
    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      detector?.close();
    };
  }, [enabled, video, retry]);

  return {
    hands,
    selectedHandIndex,
    fps,
    loading,
    error,
    retry: () => setRetry((v) => v + 1),
  };
}
