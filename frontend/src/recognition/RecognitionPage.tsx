import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { Icon } from "../Icon";
import { config, type HistoryEntry, type Prediction } from "./config";
import { useWebcam } from "./useWebcam";
import { useHandLandmarks, type Point } from "./useHandLandmarks";
import { usePredictor } from "./usePredictor";

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
  mirror,
  visible,
}: {
  video: RefObject<HTMLVideoElement | null>;
  hands: Point[][];
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
    if (!ctx || !visible) return;
    ctx.strokeStyle = "#5cf0b2";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#b2ffdd";
    for (const hand of hands) {
      for (const [a, b] of connections) {
        ctx.beginPath();
        ctx.moveTo(hand[a].x * el.width, hand[a].y * el.height);
        ctx.lineTo(hand[b].x * el.width, hand[b].y * el.height);
        ctx.stroke();
      }
      for (const p of hand) {
        ctx.beginPath();
        ctx.arc(p.x * el.width, p.y * el.height, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      const xs = hand.map((p) => p.x * el.width),
        ys = hand.map((p) => p.y * el.height);
      ctx.setLineDash([7, 5]);
      ctx.strokeRect(
        Math.min(...xs) - 12,
        Math.min(...ys) - 12,
        Math.max(...xs) - Math.min(...xs) + 24,
        Math.max(...ys) - Math.min(...ys) + 24,
      );
      ctx.setLineDash([]);
    }
  }, [video, hands, visible]);
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
  return (
    <div className="panel camera-panel">
      <div className="panel-heading">
        <h2>
          <Icon name="camera" size={18} />
          Camera trực tiếp
        </h2>
        <span className="demo-tag">
          {config.mode === "api" ? "Backend API" : "TensorFlow.js"} ·{" "}
          {config.input === "image" ? "Hình ảnh" : "Landmarks"}
        </span>
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
          mirror={mirror}
          visible={showLandmarks && camera.active && !paused}
        />
        <div className="video-top">
          <span className="fps-badge">
            <span />
            {fps} FPS · {latency === null ? "—" : latency} ms
          </span>
          <span
            className={`status-chip ${hands.length && !paused && !error ? "status-active" : ""}`}
          >
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
              {error ? "Cần kiểm tra kết nối" : loading ? "Đang tải…" : "Sẵn sàng để bắt đầu?"}
            </h3>
            <p>
              {error ||
                (loading
                  ? "Đang chuẩn bị camera và mô hình nhận diện."
                  : "Cho phép truy cập camera để nhận diện ký hiệu theo thời gian thực.")}
            </p>
            {!loading && (
              <button
                className="button"
                onClick={error ? retry : () => camera.start(camera.device)}
              >
                <Icon name="camera" size={18} />
                {error ? "Thử lại" : "Cho phép truy cập camera"}
              </button>
            )}
          </div>
        )}
        {camera.active && !loading && !error && !hands.length && !paused && (
          <div className="video-caption">Hãy đưa tay vào khung hình</div>
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
            aria-label="Lật gương"
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
      <div className="camera-footnote">
        <Icon name="shield" size={14} />
        <span>
          {config.mode === "api"
            ? `Gửi ${config.input === "image" ? "ảnh JPEG" : "điểm mốc bàn tay"} đến máy chủ model để dự đoán.`
            : "Video và nhận diện được xử lý trên thiết bị."}
        </span>
      </div>
    </div>
  );
}
export function TopKList({ prediction }: { prediction: Prediction | null }) {
  return (
    <>
      <div className="top-heading">
        TOP 3 DỰ ĐOÁN<span>Độ tin cậy</span>
      </div>
      <div className="top-predictions">
        {prediction ? (
          [...prediction.top_k]
            .sort((a, b) => b.confidence - a.confidence)
            .slice(0, 3)
            .map((p, i) => (
              <div className={`top-row ${i === 0 ? "top-first" : ""}`} key={p.label}>
                <span className="rank">{i + 1}</span>
                <span className="top-label">{p.label}</span>
                <span className="mini-track">
                  <span style={{ width: `${p.confidence * 100}%` }} />
                </span>
                <span className="top-percent">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
            ))
        ) : (
          <p className="empty-history">Chưa có dự đoán từ model.</p>
        )}
      </div>
    </>
  );
}
export function PredictionCard({
  prediction,
  threshold,
}: {
  prediction: Prediction | null;
  threshold: number;
}) {
  const confidence = prediction?.confidence || 0;
  const color = confidence < threshold ? "#94a3b8" : confidence >= 0.8 ? "#10b981" : "#b77908";
  return (
    <section className="panel prediction-panel">
      <div className="panel-heading">
        <h2>
          <Icon name="scan" size={18} />
          Kết quả hiện tại
        </h2>
        <span className="live-label">{prediction ? "MODEL" : "ĐANG CHỜ"}</span>
      </div>
      <div className="current-prediction" aria-live="polite">
        <span className="recognized-label">{prediction?.label || "—"}</span>
        <span className="recognized-hint">
          {!prediction
            ? "Kết quả sẽ xuất hiện tại đây"
            : confidence < threshold
              ? "Chưa chắc chắn — hãy thực hiện lại ký hiệu"
              : "Giữ ký hiệu ổn định để thêm vào câu"}
        </span>
      </div>
      <div className="confidence-label">
        <span>Độ tin cậy</span>
        <strong style={{ color }}>{prediction ? `${(confidence * 100).toFixed(1)}%` : "—"}</strong>
      </div>
      <div className="confidence-track">
        <span style={{ width: `${confidence * 100}%`, background: color }} />
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
          Câu đang ghép
        </h2>
        <span className="word-count">
          {sentence.trim() ? sentence.trim().split(/\s+/).length : 0} từ
        </span>
      </div>
      <textarea
        value={sentence}
        onChange={(e) => setSentence(e.target.value)}
        aria-label="Câu đang ghép"
        placeholder="Câu chuyện của bạn bắt đầu từ đây…"
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
              .then(() => notify("Đã sao chép câu."))
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
              title="Thêm vào câu"
            >
              <span className="history-check">
                <Icon name="check" size={12} />
              </span>
              <strong>{p.label}</strong>
              <span className="history-confidence">{(p.confidence * 100).toFixed(1)}%</span>
              <time>{p.time}</time>
            </button>
          ))
        ) : (
          <p className="empty-history">Chưa có ký hiệu được chấp nhận.</p>
        )}
      </div>
      <div className="history-bottom">Chọn một từ để thêm vào câu · Chỉ lưu trong phiên này</div>
    </section>
  );
}
export function SettingsDrawer({
  threshold,
  setThreshold,
  stability,
  setStability,
  landmarks,
  setLandmarks,
}: {
  threshold: number;
  setThreshold: (v: number) => void;
  stability: number;
  setStability: (v: number) => void;
  landmarks: boolean;
  setLandmarks: (v: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
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
            Ngưỡng tin cậy <strong>{Math.round(threshold * 100)}%</strong>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(+e.target.value)}
            />
          </label>
          <label>
            Số dự đoán ổn định liên tiếp
            <input
              type="number"
              min="1"
              max="60"
              value={stability}
              onChange={(e) => setStability(Math.max(1, Math.min(60, +e.target.value)))}
            />
          </label>
          <div className="toggle-row">
            <span>Hiển thị điểm mốc</span>
            <button
              className={`toggle ${landmarks ? "on" : ""}`}
              role="switch"
              aria-label="Hiển thị điểm mốc"
              aria-checked={landmarks}
              onClick={() => setLandmarks(!landmarks)}
            >
              <span />
            </button>
          </div>
          <p className="settings-note">
            {config.windowSize} khung hình · {config.requestsPerSecond} yêu cầu/giây ·{" "}
            {config.mode === "api" ? config.apiUrl : config.modelUrl}
            <br />
            Đổi chế độ và định dạng đầu vào trong src/recognition/config.ts hoặc biến môi trường
            VITE_*.
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
  const [mirror, setMirror] = useState(true),
    [paused, setPaused] = useState(false),
    [landmarks, setLandmarks] = useState(true),
    [sheet, setSheet] = useState(true);
  const [threshold, setThreshold] = useState(config.threshold),
    [stability, setStability] = useState(config.stability),
    [sentence, setSentence] = useState(""),
    [history, setHistory] = useState<HistoryEntry[]>([]);
  const touch = useRef(0),
    nextId = useRef(0);
  const append = (label: string) =>
    setSentence((s) => `${s}${s && !s.endsWith(" ") ? " " : ""}${label}`);
  const predictor = usePredictor(
    camera.video,
    camera.active && !paused,
    threshold,
    stability,
    (p) => {
      append(p.label);
      setHistory((h) =>
        [{ ...p, id: ++nextId.current, time: new Date().toLocaleTimeString("vi-VN") }, ...h].slice(
          0,
          100,
        ),
      );
    },
  );
  const detection = useHandLandmarks(camera.video, camera.active && !paused, predictor.predict);
  const error = camera.error || detection.error || (camera.active ? predictor.error : "");
  const loading = camera.loading || detection.loading || predictor.loading;
  const status = !camera.active
    ? "Đang chờ"
    : !detection.hands.length
      ? "Không thấy bàn tay"
      : "Đang nhận diện";
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
            KẾT NỐI KHÔNG GIỚI HẠN
          </div>
          <h1>
            Mỗi cử chỉ, một lời nói<span className="blue-text">.</span>
          </h1>
          <p>Biến ngôn ngữ ký hiệu thành lời nói. Kết nối theo cách của bạn.</p>
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
              if (!camera.active) void camera.start(camera.device);
            }}
          />
          <div className="tip-card">
            <span className="tip-icon">
              <Icon name="spark" />
            </span>
            <div>
              <h3>Một chút lưu ý, kết quả tốt hơn</h3>
              <p>Đảm bảo đủ ánh sáng, để tay trong khung hình và giữ mỗi ký hiệu trong 1–2 giây.</p>
            </div>
          </div>
          <SettingsDrawer
            threshold={threshold}
            setThreshold={setThreshold}
            stability={stability}
            setStability={setStability}
            landmarks={landmarks}
            setLandmarks={setLandmarks}
          />
          <div className="model-status">
            <span className="green-dot" />
            {config.mode === "api" ? "Backend API" : "TensorFlow.js"}
            <span className="tiny-dot" />
            {config.windowSize} khung hình · Dữ liệu trực tiếp
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
          <PredictionCard prediction={predictor.prediction} threshold={threshold} />
          <SentenceBuilder sentence={sentence} setSentence={setSentence} notify={notify} />
          <HistoryList history={history} clear={() => setHistory([])} append={append} />
        </aside>
      </div>
      <div className="demo-notice">
        <Icon name="info" size={15} />
        <span>
          Chỉ hiển thị dự đoán thật từ model đã cấu hình. Không có bàn tay → không gửi yêu cầu dự
          đoán.
        </span>
      </div>
    </main>
  );
}
