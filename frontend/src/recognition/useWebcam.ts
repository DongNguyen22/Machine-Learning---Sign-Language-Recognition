import { useCallback, useEffect, useRef, useState } from "react";
export function useWebcam() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const generation = useRef(0);
  const [active, setActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [device, setDevice] = useState("");
  const enumerate = useCallback(async () => {
    try {
      setDevices(
        (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput"),
      );
    } catch {
      /* Permissions may hide devices. */
    }
  }, []);
  const stop = useCallback(() => {
    generation.current++;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setActive(false);
    setLoading(false);
  }, []);
  const start = useCallback(
    async (id = "") => {
      stop();
      const token = generation.current;
      setError("");
      setLoading(true);
      try {
        if (!navigator.mediaDevices?.getUserMedia)
          throw new Error("Camera cần HTTPS hoặc localhost và trình duyệt hỗ trợ getUserMedia.");
        const media = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            ...(id ? { deviceId: { exact: id } } : { facingMode: "user" }),
          },
          audio: false,
        });
        if (token !== generation.current) {
          media.getTracks().forEach((t) => t.stop());
          return;
        }
        stream.current = media;
        if (video.current) {
          video.current.srcObject = media;
          await video.current.play();
        }
        if (token !== generation.current) return;
        media.getVideoTracks()[0].onended = () => {
          stop();
          setError("Camera đã ngắt kết nối. Vui lòng chọn camera và thử lại.");
        };
        setDevice(media.getVideoTracks()[0].getSettings().deviceId || id);
        setActive(true);
        await enumerate();
      } catch (e) {
        if (token !== generation.current) return;
        stream.current?.getTracks().forEach((t) => t.stop());
        stream.current = null;
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError" || name === "SecurityError"
            ? "Quyền camera bị từ chối. Mở biểu tượng ổ khóa → Quyền trang web → Camera → Cho phép."
            : name === "NotFoundError" || name === "OverconstrainedError"
              ? "Không tìm thấy camera phù hợp. Kết nối thiết bị hoặc chọn camera khác."
              : name === "NotReadableError" || name === "AbortError"
                ? "Camera đang được ứng dụng khác sử dụng hoặc không thể mở. Đóng ứng dụng đó và thử lại."
                : e instanceof Error
                  ? e.message
                  : "Không thể mở camera.",
        );
      } finally {
        if (token === generation.current) setLoading(false);
      }
    },
    [stop, enumerate],
  );
  useEffect(() => {
    void enumerate();
    const media = navigator.mediaDevices;
    const leave = () => stop();
    media?.addEventListener("devicechange", enumerate);
    window.addEventListener("pagehide", leave);
    return () => {
      stop();
      media?.removeEventListener("devicechange", enumerate);
      window.removeEventListener("pagehide", leave);
    };
  }, [enumerate, stop]);
  return { video, active, loading, error, devices, device, setDevice, start, stop };
}
