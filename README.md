# SignVision — Nhận diện Ngôn ngữ Ký hiệu ASL (36 lớp: `0–9, A–Z`)

Ứng dụng web nhận diện ngôn ngữ ký hiệu ASL thời gian thực sử dụng **MediaPipe 21 Hand Landmarks + Support Vector Machine (`SVC(kernel='rbf')`)** trên 36 ký hiệu (`0–9`, `A–Z`), kèm hệ thống đối chiếu định lượng tách nhóm chống rò rỉ dữ liệu và baseline **HOG + LinearSVC (Cách 3B)** được bảo toàn ở backend.

---

## 1. Kiến trúc & Mô hình hiện tại

- **Giao diện chính SignVision (`frontend/`):** chạy mặc định bằng **Landmark SVM 86 chiều** (`mediapipe_21_xy_geom_v2` — 42 tọa độ 21 điểm mốc chuẩn hóa theo cổ tay + 44 đặc trưng hình học bàn tay bất biến xoay), hỗ trợ **3 phạm vi gọn**:
  - **`Tất cả 36` (`all_36`):** 36 lớp `0–9, A–Z` — `models/svm/experiments/landmark_asl_v2/svm_landmark_asl_v2.joblib`
  - **`Chữ cái A–Z` (`letters_az`):** 26 lớp `A–Z` chuyên biệt — `models/svm/experiments/landmark_asl_v3_split/svm_landmark_az_v3.joblib`
  - **`Chữ số 0–9` (`digits_09`):** 10 lớp `0–9` chuyên biệt — `models/svm/experiments/landmark_asl_v3_split/svm_landmark_09_v3.joblib`
- **Pipeline thời gian thực độ trễ thấp:** trình duyệt chạy `@mediapipe/tasks-vision` trích xuất 21 điểm mốc trực tiếp từ webcam và gửi JSON tọa độ sang `POST /predict/landmark` (mặc định `4 Hz`, làm mượt đa số `3` khung hình, chốt ký tự sau `4` khung ổn định).
- **Baseline HOG Cách 3B ở backend:** toàn bộ 7 file model cũ trong `models/svm/` (bao gồm `models/svm/svm_final_c0_01.joblib`), endpoint `POST /predict` và bộ báo cáo đối chiếu HOG được giữ nguyên ở phía backend (ẩn khỏi UI để giao diện tinh gọn).

---

## 2. Cấu trúc thư mục chính

```text
├── frontend/                   <- Giao diện SignVision (React 19 + TypeScript + Vite 8 + MediaPipe)
├── src/
│   ├── preprocess/             <- Trích xuất đặc trưng HOG & MediaPipe 21 landmarks (42D / 86D)
│   └── model/                  <- Predictor Landmark SVM v2/v3, HOG baseline, HTTP API server
├── scripts/                    <- Huấn luyện, kiểm toán rò rỉ, thu mẫu, đánh giá webcam & kiểm thử
├── models/svm/                 <- 7 model HOG gốc + experiments/landmark_asl_{v1,v2,v3_split}/
├── data/                       <- Dữ liệu webcam đối chiếu độc lập (webcam_pilot, webcam_eval_36, ...)
├── results/svm/experiments/    <- Báo cáo JSON/CSV & báo cáo tổng kết Hướng A
└── requirements.txt            <- Thư viện Python cho backend & thí nghiệm
```

---

## 3. Khởi động nhanh (Backend + Frontend)

### Yêu cầu môi trường
- Python 3.11+ (đã cấu hình sẵn trong `.venv` theo `requirements.txt`)
- Node.js 20+ và `npm`

### Bước 1: Chạy Backend API (cổng mặc định `8000`)

Trên Windows PowerShell tại thư mục gốc repo:

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
.\.venv\Scripts\python.exe src\model\svm_api_server.py
```

Backend lắng nghe tại `http://127.0.0.1:8000` (có thể đổi bằng `SVM_API_HOST`, `SVM_API_PORT`) với các endpoint chính:
- `GET /health` — kiểm tra trạng thái model Landmark SVM (3 phạm vi) & HOG baseline
- `POST /predict/landmark` — nhận diện 21 điểm mốc bàn tay (`scope`: `all_36` | `letters_az` | `digits_09`)
- `POST /predict` — baseline HOG Cách 3B (giữ nguyên cho đối chiếu)
- `GET /labels`, `GET /reference-samples`, `POST /diagnostic/save`

### Bước 2: Chạy Frontend SignVision (cổng mặc định `8443`)

Theo cấu hình trong `frontend/vite.config.ts`, Vite dev server chạy ở cổng **`8443`** (`strictPort: true`, có thể đổi qua biến môi trường `PORT`):

```powershell
cd frontend
npm install   # chỉ cần chạy lần đầu
npm run dev
```

Mở trình duyệt tại **`http://localhost:8443`**, cấp quyền webcam và chọn phạm vi **`Tất cả 36`**, **`Chữ cái A–Z`** hoặc **`Chữ số 0–9`** để thử trực tiếp.

---

## 4. Tóm tắt kết quả đối chiếu chính

Toàn bộ quá trình kiểm toán rò rỉ dữ liệu (`asl_processed`), huấn luyện chọn trên `validation (P8, P10)`, đánh giá trên `test (P2, P7)` và đo thực tế trên webcam được ghi chi tiết tại:  
**👉 [Báo cáo tổng kết Hướng A (`LANDMARK_SVM_SUMMARY_REPORT.md`)](results/svm/experiments/LANDMARK_SVM_SUMMARY_REPORT.md)**

### Đối chiếu trên tập kiểm thử webcam độc lập đủ 36 lớp (`data/webcam_eval_36`, `77` mẫu đã xác nhận nhãn)

| Phạm vi đánh giá | Mẫu (`phát hiện/tổng`) | **HOG Cách 3B** (E2E / Cond / Macro-F1) | **Landmark `all_36`** (E2E / Cond / Macro-F1) | **Landmark chuyên biệt `A–Z` / `0–9`** (E2E / Cond / Macro-F1) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng 36 lớp (`0–9, A–Z`)** | `76 / 77` (`98.70%`) | `32.47%` / `32.89%` / `0.2954` | `61.04%` / `61.84%` / `0.5827` | **`63.64%` / `64.47%` / `0.5958`** |
| **Chữ cái `A–Z` (26 lớp)** | `52 / 52` (`100.00%`) | `30.77%` / `30.77%` / `0.2876` | `61.54%` / `61.54%` / `0.6231` | **`63.46%` / `63.46%` / `0.5987`** |
| **Chữ số `0–9` (10 lớp)** | `24 / 25` (`96.00%`) | `36.00%` / `37.50%` / `0.4105` | `60.00%` / `62.50%` / `0.5690` | **`64.00%` / `66.67%` / `0.5882`** |

> **Giới hạn khoa học cần lưu ý:** Tập kiểm thử webcam `data/webcam_eval_36` hiện chỉ gồm `77` mẫu từ **1 người chụp (`P01 / S01`)** (`2–4` mẫu/lớp), do đó kết quả trên phản ánh độ chính xác của người chụp hiện tại và **chưa đại diện cho độ tổng quát nhiều người (`≥ 3` người chụp)**. Hai ký hiệu động `J` và `Z` vẫn bị giới hạn với bộ phân loại ảnh tĩnh 1 khung hình (`J → I`).

---

## 5. Chạy bộ kiểm thử tự động

### Kiểm thử Python (10 bộ kiểm thử, không tạo `__pycache__`)

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
$tests = @(
  "scripts/test_webcam_eval_36.py",
  "scripts/test_landmark_v2_api.py",
  "scripts/test_asl_landmark_phase3_split.py",
  "scripts/test_asl_landmark_phase2b.py",
  "scripts/test_asl_landmark_phase2.py",
  "scripts/test_asl_landmark_phase1.py",
  "scripts/test_diagnostic_storage.py",
  "scripts/test_collection_flow.py",
  "scripts/test_webcam_split.py",
  "scripts/test_webcam_svm_pipeline.py"
)
foreach ($t in $tests) { .\.venv\Scripts\python.exe $t }
```

### Kiểm thử & đóng gói Frontend (3 bước)

```powershell
node --test frontend/src/recognition/predictionLogic.test.mjs
cd frontend
npx tsc --noEmit
npm run build
```
