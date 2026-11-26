import { useEffect, useState } from "react";
import { Icon, type IconName } from "./Icon";
import { RecognitionPage } from "./recognition/RecognitionPage";
import { useModelLabels } from "./recognition/usePredictor";
import { getSignInfo, SIGN_CATEGORIES, signImageUrl } from "./signInfo";
const nav = [
  { id: "home", label: "Trang chủ" },
  { id: "recognition", label: "Nhận diện" },
  { id: "dictionary", label: "Từ điển" },
  { id: "about", label: "Giới thiệu" },
];
function HandDrawing({ pose = "open" }: { pose?: string }) {
  return (
    <svg
      className={`hand-drawing ${pose}`}
      viewBox="0 0 150 160"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path
        d={
          pose === "fist"
            ? "M48 138v-28c-20-17-19-40-8-40h8V47q0-15 12-12 9-13 20-3 14-5 20 7 16-2 16 13v49q0 22-16 36"
            : pose === "one" || pose === "two"
              ? `M48 138v-28C20 82 22 72 30 69q7-4 21 13V26q0-15 12-12 6 1 6 12v47${pose === "two" ? "V20q0-13 11-12 8 1 8 14v52" : "q16-12 20 3"}q17-10 22 5 16-1 16 14v10q0 25-16 33`
              : "M48 138v-28 0C22 86 19 74 27 69q8-5 25 15V36q0-14 10-14t10 14v38-55q0-12 10-12t10 12v55-47q0-12 9-12t9 12v50-35q0-12 9-12t9 12v54q0 27-18 42"
        }
      />
      <path d="M48 138h54M56 150h40M66 96q16-9 30 1" />
      <circle cx="80" cy="100" r="3" fill="currentColor" stroke="none" />
    </svg>
  );
}

/** Ảnh thật của ký hiệu (public/images/signs/<label>.png); thiếu ảnh thì dùng hình vẽ minh họa. */
function SignImage({ label, pose }: { label: string; pose: string }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [label]);
  if (failed) return <HandDrawing pose={pose} />;
  return (
    <img
      className="sign-photo"
      src={signImageUrl(label)}
      alt={`Ký hiệu ASL chữ ${label}`}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

export default function App() {
  const [page, setPage] = useState("recognition");
  const [dark, setDark] = useState(false);
  const [menu, setMenu] = useState(false);
  const [toast, setToast] = useState("");
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("Tất cả");
  const [selectedSign, setSelectedSign] = useState<{
    label: string;
    category: string;
    description: string;
    pose: string;
  } | null>(null);
  const [help, setHelp] = useState(false);
  useEffect(() => {
    if (!help && !selectedSign) return;
    const previous = document.activeElement as HTMLElement | null;
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setHelp(false);
        setSelectedSign(null);
      }
      if (event.key !== "Tab") return;
      const items = document.querySelectorAll<HTMLElement>(".modal button, .modal a, .modal input");
      const first = items[0],
        last = items[items.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("keydown", handleKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKey);
      document.body.style.overflow = overflow;
      previous?.focus();
    };
  }, [help, selectedSign]);
  const notify = (message: string) => setToast(message);
  useEffect(() => {
    if (toast) {
      const id = setTimeout(() => setToast(""), 3500);
      return () => clearTimeout(id);
    }
  }, [toast]);
  const navigate = (id: string) => {
    setPage(id);
    setMenu(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const { labels, error: labelsError } = useModelLabels();
  const signs = labels.map((label) => ({ label, ...getSignInfo(label) }));
  const filtered = signs.filter(
    (s) =>
      (category === "Tất cả" || s.category === category) &&
      s.label.toLocaleLowerCase("vi").includes(search.toLocaleLowerCase("vi")),
  );
  return (
    <div className={`app ${dark ? "dark" : ""}`}>
      <header className="navbar">
        <a
          href="#recognition"
          className="brand"
          aria-label="SignVision - Nhận diện"
          onClick={(e) => {
            e.preventDefault();
            navigate("recognition");
          }}
        >
          <span className="brand-icon">
            <Icon name="hand" size={25} />
          </span>
          <span>
            Sign<span className="brand-blue">Vision</span>
            <span className="brand-dot">.</span>
          </span>
        </a>
        <nav className={menu ? "nav-links mobile-open" : "nav-links"} aria-label="Điều hướng chính">
          {nav.map((item) => (
            <button
              className={page === item.id ? "nav-link selected" : "nav-link"}
              aria-current={page === item.id ? "page" : undefined}
              key={item.id}
              onClick={() => navigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="nav-actions">
          <span className="language">
            <span className="language-dot" />
            Tiếng Việt
          </span>
          <span className="nav-divider" />
          <button
            className="icon-button theme-button"
            aria-label={dark ? "Bật giao diện sáng" : "Bật giao diện tối"}
            onClick={() => setDark(!dark)}
          >
            <Icon name={dark ? "sun" : "moon"} />
          </button>
          <button
            className="button nav-start"
            onClick={() => {
              navigate("recognition");
            }}
          >
            Bắt đầu ngay <Icon name="arrow" size={16} />
          </button>
          <button
            className="icon-button mobile-menu"
            aria-label="Mở menu"
            onClick={() => setMenu(!menu)}
          >
            <Icon name="menu" />
          </button>
        </div>
      </header>

      {page === "recognition" && <RecognitionPage notify={notify} onHelp={() => setHelp(true)} />}

      {page === "home" && (
        <main className="main landing-page">
          <section className="hero">
            <div className="hero-copy">
              <div className="eyebrow">
                <span />
                CÔNG NGHỆ VÌ SỰ THẤU HIỂU
              </div>
              <h1>
                Nhận diện ngôn ngữ
                <br />
                ký hiệu <span className="blue-text">bằng AI.</span>
              </h1>
              <p>
                Để mọi cử chỉ đều được lắng nghe. Chuyển ngôn ngữ ký hiệu thành văn bản và giọng nói
                theo thời gian thực, chỉ với một chiếc webcam.
              </p>
              <div className="hero-actions">
                <button className="button" onClick={() => navigate("recognition")}>
                  Bắt đầu nhận diện
                  <Icon name="arrow" size={18} />
                </button>
                <button className="button button-white" onClick={() => navigate("dictionary")}>
                  Xem danh sách ký hiệu
                  <Icon name="book" size={18} />
                </button>
              </div>
              <div className="hero-note">
                <Icon name="shield" size={16} />
                Không cần cài đặt. Tôn trọng quyền riêng tư.
              </div>
            </div>
            <div className="hero-visual">
              <img src="/images/sign-demo.png" alt="Bàn tay thể hiện ký hiệu xin chào" />
              <div className="hero-floating">
                <span className="hero-check">
                  <Icon name="check" />
                </span>
                <span>
                  <small>MỘT CỬ CHỈ, MỘT KẾT NỐI</small>
                  <strong>Mỗi cử chỉ, một kết nối</strong>
                </span>
                <Icon name="volume" />
              </div>
              <span className="hero-demo">Hình ảnh minh họa</span>
            </div>
          </section>
          <section className="features">
            {[
              {
                icon: "scan" as IconName,
                title: "Thời gian thực",
                text: "Từ cử chỉ đến văn bản trong chớp mắt. Giao tiếp tự nhiên, không gián đoạn.",
              },
              {
                icon: "spark" as IconName,
                title: "Model do nhóm tự huấn luyện",
                text: "Hướng tới mô hình được nghiên cứu và tối ưu riêng cho ngôn ngữ ký hiệu.",
              },
              {
                icon: "volume" as IconName,
                title: "Ghép câu & đọc thành tiếng",
                text: "Kết nối từng ký hiệu thành câu hoàn chỉnh và để giọng nói thay bạn sẻ chia.",
              },
            ].map((f) => (
              <article className="panel feature" key={f.title}>
                <span className="feature-icon">
                  <Icon name={f.icon} size={25} />
                </span>
                <h2>{f.title}</h2>
                <p>{f.text}</p>
              </article>
            ))}
          </section>
          <section className="how-section">
            <div className="eyebrow">ĐƠN GIẢN VÀ DỄ TIẾP CẬN</div>
            <h2>Ba bước. Vô vàn kết nối.</h2>
            <div className="steps">
              {[
                {
                  title: "Bật camera",
                  text: "Cho phép truy cập camera trên thiết bị của bạn.",
                  icon: "camera",
                },
                {
                  title: "Thực hiện ký hiệu",
                  text: "Đưa bàn tay vào khung hình và giữ ký hiệu ổn định.",
                  icon: "hand",
                },
                {
                  title: "Nhận kết quả",
                  text: "Xem văn bản, ghép câu và đọc thành tiếng.",
                  icon: "volume",
                },
              ].map((s, i) => (
                <div className="step" key={s.title}>
                  <span className="step-number">0{i + 1}</span>
                  <Icon name={s.icon as IconName} size={28} />
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </div>
              ))}
            </div>
          </section>
        </main>
      )}

      {page === "dictionary" && (
        <main className="main dictionary-page">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span />
                HỌC MỘT KÝ HIỆU, MỞ MỘT KẾT NỐI
              </div>
              <h1>
                Từ điển ký hiệu<span className="blue-text">.</span>
              </h1>
              <p>Khám phá những cử chỉ nhỏ mang ý nghĩa lớn.</p>
            </div>
            <span className="dictionary-count">{labels.length} ký hiệu từ model</span>
          </div>
          <div className="dictionary-banner">
            <Icon name="info" size={19} />
            <p>
              {labelsError ||
                "Danh sách được đọc trực tiếp từ labels.json của model. Bấm vào thẻ để xem cách tạo dáng tay."}
            </p>
          </div>
          <div className="dictionary-filters">
            <label className="search-field">
              <Icon name="search" />
              <input
                placeholder="Tìm kiếm ký hiệu…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Tìm kiếm ký hiệu"
              />
              {search && (
                <button aria-label="Xóa tìm kiếm" onClick={() => setSearch("")}>
                  <Icon name="close" size={16} />
                </button>
              )}
            </label>
            <div className="category-filters">
              {SIGN_CATEGORIES.map((c) => (
                <button
                  className={category === c ? "active" : ""}
                  aria-pressed={category === c}
                  key={c}
                  onClick={() => setCategory(c)}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <div className="sign-grid">
            {filtered.map((sign) => (
              <button
                className="panel sign-card"
                key={sign.label}
                onClick={() => setSelectedSign(sign)}
              >
                <div className="sign-illustration">
                  <SignImage label={sign.label} pose={sign.pose} />
                  <span className="sign-arrow">
                    <Icon name="arrow" size={18} />
                  </span>
                </div>
                <div className="sign-card-copy">
                  <span className="sign-category">{sign.category}</span>
                  <h2>{sign.label}</h2>
                  <p>{sign.description}</p>
                </div>
              </button>
            ))}
          </div>
          {!filtered.length && (
            <div className="search-empty">
              <Icon name="search" size={35} />
              <h2>Chưa tìm thấy ký hiệu</h2>
              <p>Thử một từ khóa hoặc danh mục khác nhé.</p>
              <button
                className="button button-white"
                onClick={() => {
                  setSearch("");
                  setCategory("Tất cả");
                }}
              >
                Xóa bộ lọc
              </button>
            </div>
          )}
        </main>
      )}

      {page === "about" && (
        <main className="main about-page">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span />
                CÔNG NGHỆ CÓ TRÁCH NHIỆM
              </div>
              <h1>
                Hiểu công nghệ. Kết nối con người<span className="blue-text">.</span>
              </h1>
              <p>SignVision được xây dựng từ một niềm tin: giao tiếp là dành cho tất cả.</p>
            </div>
          </div>
          <section className="about-intro panel">
            <span className="about-icon">
              <Icon name="hand" size={48} />
            </span>
            <div>
              <h2>Một cầu nối từ những cử chỉ</h2>
              <p>
                SignVision là giao diện thử nghiệm cho công cụ chuyển ngôn ngữ ký hiệu thành văn
                bản. Chúng tôi hướng tới trải nghiệm dễ tiếp cận, bảo vệ quyền riêng tư và mang lại
                nhiều cách giao tiếp hơn cho mọi người.
              </p>
              <span className="demo-tag">Đang trong giai đoạn nguyên mẫu</span>
            </div>
          </section>
          <div className="about-grid">
            <section className="panel about-card">
              <h2>
                <Icon name="spark" />
                Về mô hình AI
              </h2>
              <dl>
                <div>
                  <dt>Kiến trúc dự kiến</dt>
                  <dd>MediaPipe Hands + LSTM</dd>
                </div>
                <div>
                  <dt>Ngôn ngữ hướng tới</dt>
                  <dd>Ngôn ngữ ký hiệu Việt Nam (VSL)</dd>
                </div>
                <div>
                  <dt>Tập dữ liệu</dt>
                  <dd>Chưa được cung cấp</dd>
                </div>
                <div>
                  <dt>Số lượng lớp (N)</dt>
                  <dd>{labels.length} ký hiệu</dd>
                </div>
                <div>
                  <dt>Độ chính xác kiểm định</dt>
                  <dd>Chưa có kết quả đánh giá</dd>
                </div>
                <div>
                  <dt>Trạng thái tích hợp</dt>
                  <dd>
                    <span className="amber-text">Theo cấu hình API / TensorFlow.js</span>
                  </dd>
                </div>
              </dl>
            </section>
            <section className="panel about-card limitations">
              <h2>
                <Icon name="info" />
                Hạn chế & lưu ý
              </h2>
              <ul>
                <li>
                  Dự đoán phụ thuộc model do nhóm cung cấp. Điểm tin cậy không phải độ chính xác đã
                  được kiểm định độc lập.
                </li>
                <li>
                  VSL và ASL là những ngôn ngữ khác nhau. Bản trải nghiệm định hướng VSL, chưa hỗ
                  trợ dịch ASL.
                </li>
                <li>
                  Ánh sáng, góc máy và tốc độ cử động có thể ảnh hưởng đến chất lượng nhận diện.
                </li>
                <li>
                  Không thay thế phiên dịch viên trong các tình huống y tế, pháp lý hoặc khẩn cấp.
                </li>
                <li>Ngôn ngữ ký hiệu có sự khác biệt theo vùng miền và ngữ cảnh.</li>
              </ul>
            </section>
          </div>
          <section className="team-section">
            <div className="eyebrow">NHỮNG NGƯỜI ĐỨNG SAU KẾT NỐI</div>
            <h2>Được xây dựng bằng sự thấu hiểu</h2>
            <p>
              Thông tin thành viên và đơn vị nghiên cứu sẽ được bổ sung khi dự án chính thức ra mắt.
            </p>
            <div className="team-roles">
              {["Nghiên cứu AI", "Phát triển sản phẩm", "Thiết kế trải nghiệm"].map((r, i) => (
                <div className="panel team-role" key={r}>
                  <span>0{i + 1}</span>
                  <h3>{r}</h3>
                  <p>Thông tin đang cập nhật</p>
                </div>
              ))}
            </div>
          </section>
        </main>
      )}

      <footer className="footer">
        <span className="footer-brand">
          <Icon name="hand" size={17} />
          SignVision<span className="footer-separator">|</span>
          <span>Kết nối từ những điều không lời.</span>
        </span>
        <span>
          Được tạo ra để mọi người gần nhau hơn.
          <span className="footer-dot" />
        </span>
      </footer>
      {toast && (
        <div className="toast" role="status">
          <span className="toast-check">
            <Icon name="check" size={17} />
          </span>
          {toast}
          <button className="icon-button" aria-label="Đóng thông báo" onClick={() => setToast("")}>
            <Icon name="close" size={15} />
          </button>
        </div>
      )}
      {(help || selectedSign) && (
        <div
          className="modal-backdrop"
          onClick={() => {
            setHelp(false);
            setSelectedSign(null);
          }}
        >
          <section
            className="modal panel"
            role="dialog"
            aria-modal="true"
            aria-label={selectedSign ? selectedSign.label : "Hướng dẫn sử dụng"}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="icon-button modal-close"
              aria-label="Đóng"
              onClick={() => {
                setHelp(false);
                setSelectedSign(null);
              }}
            >
              <Icon name="close" />
            </button>
            {selectedSign ? (
              <>
                <div className="modal-sign">
                  <SignImage label={selectedSign.label} pose={selectedSign.pose} />
                </div>
                <span className="sign-category">{selectedSign.category}</span>
                <h2>{selectedSign.label}</h2>
                <p>{selectedSign.description}</p>
                <p className="modal-note">
                  Hình minh họa tham khảo bảng chữ cái ASL (American Sign Language). Đặt bàn tay
                  rõ trong khung hình camera để model nhận diện tốt nhất.
                </p>
                <button
                  className="button"
                  onClick={() => {
                    setSelectedSign(null);
                    navigate("recognition");
                  }}
                >
                  Thử với camera
                  <Icon name="arrow" size={17} />
                </button>
              </>
            ) : (
              <>
                <span className="feature-icon">
                  <Icon name="hand" size={28} />
                </span>
                <h2>Cùng bắt đầu nhé!</h2>
                <p>Chỉ vài bước nhỏ để khám phá SignVision.</p>
                <ol className="help-steps">
                  <li>
                    <strong>Cho phép sử dụng camera</strong>
                    <p>Nhấn “Bật camera” và cho phép trình duyệt truy cập thiết bị.</p>
                  </li>
                  <li>
                    <strong>Chuẩn bị khung hình</strong>
                    <p>Chọn nơi đủ sáng, đưa bàn tay vào camera và giữ ổn định.</p>
                  </li>
                  <li>
                    <strong>Khám phá bản demo</strong>
                    <p>
                      Giữ cùng một ký hiệu đến khi đủ độ tin cậy và số dự đoán ổn định. Kết quả được
                      tự động thêm vào câu.
                    </p>
                  </li>
                </ol>
                <button className="button" onClick={() => setHelp(false)}>
                  Đã hiểu
                  <Icon name="check" size={17} />
                </button>
              </>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
