import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { Icon } from "../Icon";
import {
  config,
  LANDMARK_SCOPES,
  type HistoryEntry,
  type LandmarkScope,
  type Prediction,
} from "./config";
import { resolveLandmarkScope } from "./predictionLogic";
import { useWebcam } from "./useWebcam";
import { useHandLandmarks, type Point } from "./useHandLandmarks";
import { usePredictor, type BackendHealth } from "./usePredictor";

type Camera = ReturnType<typeof useWebcam>;
const connections = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
];

export function LandmarkOverlay({
  video,
  hands,
  selectedHandIndex,
  mirror,
  visible,
}: {
  video: RefObject<HTMLVideoElement | null>;
  hands: Point[][];
  selectedHandIndex: number;
  mirror: boolean;
  visible: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const el = canvas.current,
      source = video.current;
    if (!el || !source) return;
    el.width = source.videoWidth || 640;
    el.height = source.videoHeight || 480;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, el.width, el.height);
    if (!visible) return;

    hands.forEach((hand, idx) => {
      const isSelected = idx === selectedHandIndex || hands.length === 1;
      ctx.strokeStyle = isSelected ? "#5cf0b2" : "rgba(148, 163, 184, 0.55)";
      ctx.lineWidth = isSelected ? 2.2 : 1.4;
      ctx.fillStyle = isSelected ? "#b2ffdd" : "rgba(203, 213, 225, 0.55)";

      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * el.width, hand[a].y * el.height);
        ctx.lineTo(hand[b].x * el.width, hand[b].y * el.height);
        ctx.stroke();
      }
      for (const p of hand) {
        ctx.beginPath();
        ctx.arc(p.x * el.width, p.y * el.height, isSelected ? 4 : 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }, [video, hands, selectedHandIndex, visible]);

  return (
    <canvas
      ref={canvas}
      className={`real-landmarks ${mirror ? "mirrored" : ""}`}
      aria-hidden="true"
    />
  );
}

export function WebcamView({
  camera,
  hands,
  selectedHandIndex,
  activeScope,
  onSelectScope,
  backend,
  mirror,
  setMirror,
  showLandmarks,
  paused,
  setPaused,
  fps,
  latency,
  status,
  error,
  retry,
  loading,
}: {
  camera: Camera;
  hands: Point[][];
  selectedHandIndex: number;
  activeScope: LandmarkScope | null;
  onSelectScope: (scope: LandmarkScope) => void;
  backend: BackendHealth;
  mirror: boolean;
  setMirror: (v: boolean) => void;
  showLandmarks: boolean;
  paused: boolean;
  setPaused: (v: boolean) => void;
  fps: number;
  latency: number | null;
  status: string;
  error: string;
  retry: () => void;
  loading: boolean;
}) {
  const frame = useRef<HTMLDivElement>(null);
  const takePhoto = () => {
    const v = camera.video.current;
    if (!v?.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    const ctx = c.getContext("2d")!;
    if (mirror) {
      ctx.translate(c.width, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(v, 0, 0);
    const link = document.createElement("a");
    link.href = c.toDataURL("image/png");
    link.download = "signvision.png";
    link.click();
  };

  const isRecognizing = camera.active && !paused && !error && hands.length > 0 && activeScope !== null;

  return (
    <div className="panel camera-panel">
      <div className="panel-heading" style={{ gap: "0.6rem", flexWrap: "wrap" }}>
        <h2>
          <Icon name="camera" size={18} />
          Camera trực tiếp
        </h2>
        <div
          role="group"
          aria-label="Chọn phạm vi nhận diện Landmark SVM"
          style={{ display: "inline-flex", gap: "0.35rem", flexWrap: "wrap", alignItems: "center" }}
        >
          {LANDMARK_SCOPES.map((item) => {
            const avail = backend.scopes[item.id]?.available ?? false;
            const selected = activeScope === item.id;
            return (
              <button
                key={item.id}
                type="button"
                className={`control-button mirror-button ${selected ? "control-selected" : ""}`}
                aria-pressed={selected}
                disabled={backend.status === "ready" ? !avail : false}
                onClick={() => onSelectScope(item.id)}
                title={
                  avail || backend.status !== "ready"
                    ? item.description
                    : backend.scopes[item.id]?.error || `Phạm vi ${item.label} chưa sẵn sàng trên máy chủ`
                }
                style={{ padding: "0.32rem 0.68rem", fontSize: "0.8rem" }}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
      <div className="webcam real-webcam" ref={frame}>
        <video
          className={`camera-media ${mirror ? "mirrored" : ""}`}
          ref={camera.video}
          autoPlay
          muted
          playsInline
        />
        <LandmarkOverlay
          video={camera.video}
          hands={hands}
          selectedHandIndex={selectedHandIndex}
          mirror={mirror}
          visible={showLandmarks && camera.active && !paused}
        />
        <div className="video-top">
          <span className="fps-badge">
            <span />
            {fps} FPS · {latency === null ? "—" : `${latency} ms`}
          </span>
          <span className={`status-chip ${isRecognizing ? "status-active" : ""}`}>
            <span />
            {paused ? "Đã tạm dừng" : status}
          </span>
        </div>
        {(!camera.active || loading || error) && (
          <div className="camera-state">
            <span className={`state-icon ${loading ? "loading-icon" : ""}`}>
              <Icon name={error ? "info" : "camera"} size={34} />
            </span>
            <h3>
              {error
                ? "Cần kiểm tra kết nối hoặc thiết bị"
                : loading
                  ? "Đang kết nối…"
                  : "Sẵn sàng để bắt đầu?"}
            </h3>
            <p>
              {error ||
                (loading
                  ? "Đang kiểm tra máy chủ Landmark SVM và bộ phát hiện bàn tay MediaPipe."
                  : "Nhấn Bật camera và giơ một bàn tay vào khung hình để nhận diện ký hiệu ASL (chọn Tất cả 36, Chữ cái A–Z hoặc Chữ số 0–9).")}
            </p>
            {!loading && (
              <div
                style={{
                  display: "flex",
                  gap: "0.6rem",
                  flexWrap: "wrap",
                  justifyContent: "center",
                }}
              >
                <button
                  className="button"
                  onClick={error ? retry : () => camera.start(camera.device)}
                >
                  <Icon name="camera" size={18} />
                  {error ? "Thử lại" : "Bật camera"}
                </button>
              </div>
            )}
          </div>
        )}
        {camera.active && !loading && !error && !paused && !hands.length && (
          <div className="video-caption">Chưa phát hiện bàn tay — Hãy đưa bàn tay vào khung hình</div>
        )}
        {camera.active && !loading && !error && !paused && hands.length > 1 && (
          <div className="video-caption">
            Phát hiện {hands.length} bàn tay — Đang theo dõi bàn tay lớn nhất
          </div>
        )}
        <button
          className="video-expand"
          aria-label="Toàn màn hình"
          onClick={() => {
            if (document.fullscreenElement) void document.exitFullscreen();
            else void frame.current?.requestFullscreen().catch(() => {});
          }}
        >
          <Icon name="expand" size={18} />
        </button>
      </div>
      <div className="camera-controls">
        <div className="control-left">
          <button
            className={`button camera-toggle ${camera.active ? "button-stop" : ""}`}
            disabled={camera.loading}
            onClick={() => (camera.active ? camera.stop() : camera.start(camera.device))}
            aria-label={camera.active ? "Tắt camera" : "Bật camera"}
          >
            <Icon name="camera" size={18} />
            <span>{camera.active ? "Tắt camera" : "Bật camera"}</span>
          </button>
          <button
            className="control-button"
            disabled={!camera.active}
            aria-label={paused ? "Tiếp tục nhận diện" : "Tạm dừng nhận diện"}
            aria-pressed={paused}
            onClick={() => setPaused(!paused)}
          >
            <Icon name={paused ? "play" : "pause"} />
          </button>
          <button
            className={`control-button mirror-button ${mirror ? "control-selected" : ""}`}
            aria-label="Lật gương xem trước"
            aria-pressed={mirror}
            onClick={() => setMirror(!mirror)}
          >
            <Icon name="mirror" />
            <span>Lật gương</span>
          </button>
        </div>
        <div className="control-right">
          <label className="camera-select">
            <Icon name="camera" size={16} />
            <select
              aria-label="Chọn camera"
              value={camera.device}
              onChange={(e) => {
                camera.setDevice(e.target.value);
                if (camera.active) void camera.start(e.target.value);
              }}
            >
              <option value="">Camera mặc định</option>
              {camera.devices.map((d) => (
                <option value={d.deviceId} key={d.deviceId}>
                  {d.label || `Camera ${camera.devices.indexOf(d) + 1}`}
                </option>
              ))}
            </select>
            <Icon name="chevron" size={13} />
          </label>
          <button
            className="control-button"
            disabled={!camera.active}
            aria-label="Chụp ảnh"
            onClick={takePhoto}
          >
            <Icon name="download" />
          </button>
        </div>
      </div>
    </div>
  );
}

export function TopKList({ prediction }: { prediction: Prediction | null }) {
  const topItems = prediction ? prediction.top_k.slice(0, 3) : [];
  return (
    <>
      <div className="top-heading">
        TOP 3 DỰ ĐOÁN<span>Tỉ trọng OvR</span>
      </div>
      <div className="top-predictions">
        {prediction ? (
          topItems.map((p, i) => {
            const weight = typeof p.relative_weight === "number" ? p.relative_weight : 0;
            const barPct = Math.max(8, Math.min(100, weight * 100));
            return (
              <div className={`top-row ${i === 0 ? "top-first" : ""}`} key={p.label}>
                <span className="rank">{i + 1}</span>
                <span className="top-label">{p.label}</span>
                <span className="mini-track" title={`OvR score: ${p.decision_score.toFixed(3)}`}>
                  <span style={{ width: `${barPct}%` }} />
                </span>
                <span className="top-percent">{Math.round(weight * 100)}%</span>
              </div>
            );
          })
        ) : (
          <p className="empty-history">Chưa có dự đoán từ model.</p>
        )}
      </div>
    </>
  );
}

export function PredictionCard({
  prediction,
  cameraActive,
  hasHand,
  activeScope,
}: {
  prediction: Prediction | null;
  cameraActive: boolean;
  hasHand: boolean;
  activeScope: LandmarkScope | null;
}) {
  const scopeMeta = LANDMARK_SCOPES.find((item) => item.id === activeScope);
  const marginText = prediction ? prediction.margin.toFixed(2) : "—";
  const confidenceText = prediction ? `${Math.round(prediction.confidence * 100)}%` : "—";

  let hintText = "Bật camera và đưa bàn tay vào khung hình";
  if (cameraActive && !hasHand) {
    hintText = "Chưa phát hiện bàn tay";
  } else if (prediction) {
    hintText = "Giữ ký hiệu ổn định để tự động thêm vào chuỗi";
  }

  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <h2>
          <Icon name="scan" size={18} />
          Kết quả nhận diện
        </h2>
        <span className="live-label">
          {cameraActive && !hasHand
            ? "CHƯA CÓ TAY"
            : scopeMeta?.badge || "LANDMARK SVM"}
        </span>
      </div>
      <div className="current-prediction" aria-live="polite">
        <span className="recognized-label">{prediction?.label || "—"}</span>
        <span className="recognized-hint">{hintText}</span>
      </div>
      <div className="confidence-label">
        <span>Độ tin cậy (ước lượng OvR) · Δ</span>
        <strong style={{ color: prediction ? "#10b981" : "#94a3b8" }}>
          {prediction ? `${confidenceText} (Δ ${marginText})` : "—"}
        </strong>
      </div>
      <div className="prediction-separator" />
      <TopKList prediction={prediction} />
    </section>
  );
}

export function SentenceBuilder({
  sentence,
  setSentence,
  notify,
}: {
  sentence: string;
  setSentence: (s: string | ((s: string) => string)) => void;
  notify: (s: string) => void;
}) {
  return (
    <section className="panel sentence-panel">
      <div className="panel-heading">
        <h2>
          <Icon name="book" size={17} />
          Chuỗi ký hiệu đã ghép
        </h2>
        <span className="word-count">
          {sentence.trim() ? sentence.trim().split(/\s+/).length : 0} ký hiệu
        </span>
      </div>
      <textarea
        value={sentence}
        onChange={(e) => setSentence(e.target.value)}
        aria-label="Chuỗi ký hiệu đã ghép"
        placeholder="Các ký hiệu giữ ổn định sẽ được ghép vào đây…"
      />
      <div className="sentence-tools">
        <button onClick={() => setSentence((s) => s + " ")}>
          <Icon name="space" size={16} />
          Khoảng trắng
        </button>
        <button
          aria-label="Xóa ký tự"
          onClick={() => setSentence((s) => Array.from(s).slice(0, -1).join(""))}
        >
          <Icon name="delete" size={17} />
        </button>
        <button className="clear-button" onClick={() => setSentence("")}>
          <Icon name="trash" size={15} />
          Xóa hết
        </button>
      </div>
      <div className="sentence-actions">
        <button
          className="button"
          disabled={!sentence.trim()}
          onClick={() => {
            if (!("speechSynthesis" in window))
              return notify("Trình duyệt chưa hỗ trợ đọc văn bản.");
            speechSynthesis.cancel();
            const speech = new SpeechSynthesisUtterance(sentence);
            speech.lang = "vi-VN";
            speechSynthesis.speak(speech);
          }}
        >
          <Icon name="volume" size={18} />
          Đọc to
        </button>
        <button
          className="button button-white"
          disabled={!sentence.trim()}
          onClick={() => {
            void navigator.clipboard
              ?.writeText(sentence)
              .then(() => notify("Đã sao chép chuỗi ký hiệu."))
              .catch(() => notify("Không thể sao chép. Hãy chọn văn bản và sao chép trực tiếp."));
          }}
        >
          <Icon name="copy" size={16} />
          Sao chép
        </button>
      </div>
    </section>
  );
}

export function HistoryList({
  history,
  clear,
  append,
}: {
  history: HistoryEntry[];
  clear: () => void;
  append: (s: string) => void;
}) {
  return (
    <section className="panel history-panel">
      <div className="panel-heading">
        <h2>
          <Icon name="clock" size={17} />
          Lịch sử nhận diện
        </h2>
        <button className="text-button" disabled={!history.length} onClick={clear}>
          Xóa
        </button>
      </div>
      <div className="history-list">
        {history.length ? (
          history.map((p) => (
            <button
              className="history-row"
              key={p.id}
              onClick={() => append(p.label)}
              title="Thêm vào chuỗi ký hiệu"
            >
              <span className="history-check">
                <Icon name="check" size={12} />
              </span>
              <strong>{p.label}</strong>
              <span className="history-confidence">{Math.round(p.confidence * 100)}%</span>
              <time>{p.time}</time>
            </button>
          ))
        ) : (
          <p className="empty-history">Chưa có ký hiệu được chấp nhận.</p>
        )}
      </div>
    </section>
  );
}

export function SettingsDrawer({
  requestsPerSecond,
  setRequestsPerSecond,
  smoothingWindow,
  setSmoothingWindow,
  stability,
  setStability,
  landmarks,
  setLandmarks,
  activeScope,
  backend,
}: {
  requestsPerSecond: number;
  setRequestsPerSecond: (v: number) => void;
  smoothingWindow: number;
  setSmoothingWindow: (v: number) => void;
  stability: number;
  setStability: (v: number) => void;
  landmarks: boolean;
  setLandmarks: (v: boolean) => void;
  activeScope: LandmarkScope | null;
  backend: BackendHealth;
}) {
  const [open, setOpen] = useState(false);
  const scopeMeta = LANDMARK_SCOPES.find((item) => item.id === activeScope);
  return (
    <div className={`panel settings-panel ${open ? "is-open" : ""}`}>
      <button className="settings-heading" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span>
          <Icon name="settings" />
          <strong>Cài đặt nhận diện</strong>
        </span>
        <Icon name="chevron" />
      </button>
      {open && (
        <div className="settings-body">
          <label>
            Tốc độ dự đoán: <strong>{requestsPerSecond} khung/giây</strong>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={requestsPerSecond}
              onChange={(e) => setRequestsPerSecond(+e.target.value)}
            />
          </label>
          <label>
            Làm mượt hiển thị: <strong>{smoothingWindow} khung</strong>
            <input
              type="range"
              min="1"
              max="7"
              step="1"
              value={smoothingWindow}
              onChange={(e) => setSmoothingWindow(+e.target.value)}
            />
          </label>
          <label>
            Số khung ổn định để ghép câu
            <input
              type="number"
              min="1"
              max="30"
              value={stability}
              onChange={(e) => setStability(Math.max(1, Math.min(30, +e.target.value)))}
            />
          </label>
          <div className="toggle-row">
            <span>Hiển thị 21 điểm mốc MediaPipe</span>
            <button
              className={`toggle ${landmarks ? "on" : ""}`}
              role="switch"
              aria-label="Hiển thị 21 điểm mốc MediaPipe"
              aria-checked={landmarks}
              onClick={() => setLandmarks(!landmarks)}
            >
              <span />
            </button>
          </div>
          <p className="settings-note">
            API: {config.landmarkApiUrl} ·{" "}
            {backend.status === "ready"
              ? backend.landmarkAvailable
                ? `Sẵn sàng (${scopeMeta?.label || "Landmark SVM"} · ${backend.featureSchema || "mediapipe_21_xy_geom_v2"})`
                : "Chưa nạp được model Landmark SVM"
              : backend.status === "checking"
                ? "Đang kiểm tra…"
                : "Chưa kết nối"}
          </p>
        </div>
      )}
    </div>
  );
}

export function RecognitionPage({
  notify,
  onHelp,
}: {
  notify: (s: string) => void;
  onHelp: () => void;
}) {
  const camera = useWebcam();
  const [mirror, setMirror] = useState(true);
  const [paused, setPaused] = useState(false);
  const [landmarks, setLandmarks] = useState(true);
  const [requestedScope, setRequestedScope] = useState<LandmarkScope>("all_36");
  const [sheet, setSheet] = useState(true);
  const [requestsPerSecond, setRequestsPerSecond] = useState(config.requestsPerSecond);
  const [smoothingWindow, setSmoothingWindow] = useState(config.smoothingWindow);
  const [stability, setStability] = useState(config.stability);
  const [sentence, setSentence] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const touch = useRef(0);
  const nextId = useRef(0);

  const append = (label: string) =>
    setSentence((s) => `${s}${s && !s.endsWith(" ") ? " " : ""}${label}`);

  // Effective scope is resolved cleanly once /health reports scope availability
  const [resolvedScope, setResolvedScope] = useState<LandmarkScope | null>("all_36");

  const predictor = usePredictor(
    camera.video,
    camera.active && !paused,
    resolvedScope,
    requestsPerSecond,
    smoothingWindow,
    stability,
    (p) => {
      append(p.label);
      setHistory((h) =>
        [
          {
            label: p.label,
            decisionScore: p.decisionScore,
            confidence: p.confidence,
            margin: p.margin,
            scope: p.scope,
            id: ++nextId.current,
            time: new Date().toLocaleTimeString("vi-VN"),
          },
          ...h,
        ].slice(0, 100),
      );
    },
  );

  useEffect(() => {
    if (predictor.backend.status !== "ready") {
      setResolvedScope(requestedScope);
      return;
    }
    const next = resolveLandmarkScope(requestedScope, {
      all_36: predictor.backend.scopes.all_36.available,
      letters_az: predictor.backend.scopes.letters_az.available,
      digits_09: predictor.backend.scopes.digits_09.available,
    });
    setResolvedScope(next);
    if (next && next !== requestedScope) {
      setRequestedScope(next);
    }
  }, [
    requestedScope,
    predictor.backend.status,
    predictor.backend.scopes.all_36.available,
    predictor.backend.scopes.letters_az.available,
    predictor.backend.scopes.digits_09.available,
  ]);

  const detection = useHandLandmarks(
    camera.video,
    camera.active && !paused,
    predictor.predict,
  );

  const error =
    camera.error ||
    (predictor.backend.status === "offline"
      ? predictor.backend.error || ""
      : predictor.backend.status === "ready" && !predictor.backend.landmarkAvailable
        ? predictor.backend.error || "Chưa nạp được model Landmark SVM trên máy chủ."
        : "") ||
    detection.error ||
    (camera.active ? predictor.error : "");
  const loading = camera.loading || detection.loading || predictor.loading;

  const activeScopeMeta = LANDMARK_SCOPES.find((item) => item.id === resolvedScope);
  const status =
    predictor.backend.status === "checking"
      ? "Đang kết nối…"
      : predictor.backend.status === "offline"
        ? "Backend chưa kết nối"
        : !predictor.backend.landmarkAvailable
          ? "Chưa sẵn sàng Landmark SVM"
          : !camera.active
            ? "Camera đang tắt"
            : !detection.hands.length
              ? "Chưa phát hiện bàn tay"
              : `Đang nhận diện (${activeScopeMeta?.label || "Landmark SVM"})`;

  useEffect(
    () => () => {
      window.speechSynthesis?.cancel();
    },
    [],
  );

  return (
    <main className="main recognition-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">
            <span />
            NHẬN DIỆN KÝ HIỆU TĨNH ASL (LANDMARK SVM)
          </div>
          <h1>
            Mỗi cử chỉ, một lời nói<span className="blue-text">.</span>
          </h1>
          <p>
            Bật camera, chọn phạm vi phù hợp (Tất cả 36, Chữ cái A–Z hoặc Chữ số 0–9) và đưa bàn tay vào khung hình.
          </p>
        </div>
        <button className="button button-white guide-button" onClick={onHelp}>
          <Icon name="help" size={17} />
          Hướng dẫn sử dụng
        </button>
      </div>
      <div className="workspace">
        <section className="camera-column" aria-label="Camera nhận diện">
          <WebcamView
            camera={camera}
            hands={detection.hands}
            selectedHandIndex={detection.selectedHandIndex}
            activeScope={resolvedScope}
            onSelectScope={setRequestedScope}
            backend={predictor.backend}
            mirror={mirror}
            setMirror={setMirror}
            showLandmarks={landmarks}
            paused={paused}
            setPaused={setPaused}
            fps={detection.fps}
            latency={predictor.latency}
            status={status}
            error={error}
            loading={loading}
            retry={() => {
              predictor.retry();
              detection.retry();
              if (camera.error) {
                void camera.start(camera.device);
              }
            }}
          />
          <SettingsDrawer
            requestsPerSecond={requestsPerSecond}
            setRequestsPerSecond={setRequestsPerSecond}
            smoothingWindow={smoothingWindow}
            setSmoothingWindow={setSmoothingWindow}
            stability={stability}
            setStability={setStability}
            landmarks={landmarks}
            setLandmarks={setLandmarks}
            activeScope={resolvedScope}
            backend={predictor.backend}
          />
          <div className="model-status">
            <span
              className="green-dot"
              style={{
                background:
                  predictor.backend.status === "ready" && predictor.backend.landmarkAvailable
                    ? "#10b981"
                    : predictor.backend.status === "checking"
                      ? "#f59e0b"
                      : "#ef4444",
              }}
            />
            {predictor.backend.status === "ready"
              ? predictor.backend.landmarkAvailable
                ? `Landmark SVM sẵn sàng (${activeScopeMeta?.label || "Tất cả 36"})`
                : "Landmark SVM chưa sẵn sàng"
              : predictor.backend.status === "checking"
                ? "Đang kết nối máy chủ Landmark SVM…"
                : "Máy chủ chưa kết nối"}
            <span className="tiny-dot" />
            {requestsPerSecond} FPS · 86 đặc trưng hình học
          </div>
        </section>
        <aside className={`results-column ${sheet ? "sheet-open" : "sheet-closed"}`}>
          <button
            className="sheet-handle"
            aria-label={sheet ? "Thu gọn kết quả" : "Mở rộng kết quả"}
            onClick={() => setSheet(!sheet)}
            onTouchStart={(e) => {
              touch.current = e.touches[0].clientY;
            }}
            onTouchEnd={(e) => {
              const dy = e.changedTouches[0].clientY - touch.current;
              if (Math.abs(dy) > 20) setSheet(dy < 0);
            }}
          >
            <span />
          </button>
          <PredictionCard
            prediction={predictor.prediction}
            cameraActive={camera.active && !paused}
            hasHand={detection.hands.length > 0}
            activeScope={resolvedScope}
          />
          <SentenceBuilder sentence={sentence} setSentence={setSentence} notify={notify} />
          <HistoryList history={history} clear={() => setHistory([])} append={append} />
        </aside>
      </div>
    </main>
  );
}
