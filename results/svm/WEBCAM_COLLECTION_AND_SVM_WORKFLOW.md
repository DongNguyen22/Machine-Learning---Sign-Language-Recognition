# Quy trình Thu thập Dữ liệu Webcam & Huấn luyện SVM cho SignVision (36 lớp: 0–9, A–Z)

Tài liệu này áp dụng cấu trúc quy trình 4 bước từ video tham khảo [Sign language detection with Python and Scikit Learn](https://www.youtube.com/watch?v=MJCSjXepaAM) (`computervisioneng/sign-language-detector-python`) vào dự án **SignVision**, đồng thời giữ rõ sự khác biệt về mô hình và đặc trưng.

---

## 1. Phân biệt rõ Video tham khảo vs. Dự án SignVision

| Thành phần | Video tham khảo (`computervisioneng`) | SignVision — Phương án chính (hiện tại) | SignVision — Thí nghiệm riêng (tùy chọn) |
| :--- | :--- | :--- | :--- |
| **Thu ảnh** | `collect_imgs.py` (chụp liên tiếp 100 frame/lớp, 1 người) | Giao diện SignVision (`/diagnostic/save`) hoặc `scripts/capture_webcam_collection.py` — bắt buộc 6 trường metadata, tách lần giơ tay, cảnh báo gần trùng | Dùng chung bộ ảnh đã xác nhận nhãn thật |
| **Đặc trưng** | 21 điểm mốc MediaPipe (`x - min_x, y - min_y`) | **1764 đặc trưng HOG (`float32`)** từ ảnh crop bàn tay → `resize_with_padding(64×64)` → grayscale (hoặc Cách 3B + HOG) | **42 đặc trưng MediaPipe Landmarks (`float32`)** chuẩn hóa theo cổ tay & bán kính lớn nhất (`mediapipe_21_xy_wrist_norm_v1`) |
| **Mô hình** | `RandomForestClassifier` (chia ngẫu nhiên 80/20) | **`LinearSVC`** (36 lớp `0–9, A–Z`), lưu riêng trong `models/svm/` | **`StandardScaler + LinearSVC` / `SVC(kernel='rbf')`**, lưu riêng trong `models/svm/experiments/` |
| **Chia tập** | `train_test_split` ngẫu nhiên (dễ rò rỉ frame liền kề) | **Tách theo `participant_id`** khi đủ ≥ 3 người; cấm rò rỉ `gesture_instance_id` & trùng SHA-256 | Dùng chung đúng file `split.csv` đã kiểm chứng |

> **Nguyên tắc bắt buộc**: Không sao chép `RandomForestClassifier` rồi gọi là huấn luyện SVM; không bao giờ nạp 42 đặc trưng landmarks vào mô hình HOG 1764 chiều cũ (`models/svm/svm_final_c0_01.joblib`).

---

## 2. Bước 1 — Chụp đúng 1 mẫu thật qua Giao diện SignVision để kiểm chứng camera

1. **Mở Terminal 1** (từ thư mục gốc dự án) khởi động Python SVM API:
   ```powershell
   .\.venv\Scripts\python.exe src/model/svm_api_server.py
   ```
2. **Mở Terminal 2** khởi động Frontend SignVision:
   ```powershell
   cd frontend
   npm run dev
   ```
3. Mở trình duyệt tại `http://localhost:5173`:
   - Bấm **Bật camera** và đảm bảo **Chế độ chẩn đoán** đang BẬT.
   - Giơ rõ **1 ký hiệu ASL** (ví dụ `A`) trong khung MediaPipe.
   - Bấm **Đóng băng khung hình này**.
   - Điền đủ 6 trường metadata:
     - `participant_id`: VD `P01`
     - `session_id`: VD `S01`
     - `gesture_instance_id`: VD `A_G01`
     - `handedness`: chọn `right` hoặc `left`
     - `background`: VD `plain_wall`
     - `lighting`: VD `daylight`
   - Chọn **Ký hiệu đang giơ** (VD `A`), rồi bấm **Xác nhận khung hình và nhãn — Lưu mẫu chẩn đoán**.
4. Chạy lệnh kiểm tra tự động thư mục mẫu vừa tạo (thay `<ten_thu_muc_mau>` bằng đường dẫn UI vừa báo):
   ```powershell
   .\.venv\Scripts\python.exe scripts/verify_single_sample.py "results/svm/webcam_diagnostics/<ten_thu_muc_mau>"
   ```

---

## 3. Đợt chụp thử nhỏ (Pilot) trước khi thu hàng nghìn ảnh

Trước khi thu đủ 36 lớp từ nhiều người, hãy chạy một **đợt pilot nhỏ** vào thư mục riêng `data/webcam_pilot` (giữ nguyên 55 mẫu cũ trong `results/svm/webcam_diagnostics/`):
- **Mục tiêu pilot**: 4 lớp (`A, B, C, D`) × 3 lần giơ tay tách biệt mỗi lớp (`G01, G02, G03`) = **12 ảnh** (mỗi lần giơ phải hạ tay xuống hoàn toàn rồi giơ lại ở góc/vị trí hơi khác).
- Có thể chụp qua **Giao diện SignVision** (có cảnh báo gần trùng trực tiếp khi bấm lưu, không tự xóa/đổi nhãn) hoặc công cụ CLI dùng đúng crop MediaPipe 18% như giao diện (tự động fallback `CAP_DSHOW` trên Windows nếu MSMF lỗi):
  ```powershell
  .\.venv\Scripts\python.exe scripts/capture_webcam_collection.py `
    --output data/webcam_pilot `
    --participant-id P01 `
    --session-id S01 `
    --gesture-instance-id A_G01 `
    --handedness right `
    --background plain_wall `
    --lighting daylight `
    --label A `
    --roi-mode mediapipe `
    --count 3 `
    --auto-next-instance
  ```
- Sau khi chụp xong pilot, chạy đủ 4 bước kiểm tra **lưu → duyệt → tạo đặc trưng → chia tập (dry-run)**:
  ```powershell
  # 1. [Lưu & Tạo đặc trưng] Kiểm tra bitwise HOG 1764 + MediaPipe 42 trên 1 mẫu bất kỳ trong pilot:
  .\.venv\Scripts\python.exe scripts/verify_single_sample.py "data/webcam_pilot/<ten_thu_muc_mau>"

  # 2. [Duyệt nhãn] Tạo bảng ảnh HTML/PNG/CSV để kiểm tra mắt từng ảnh (chưa xác nhận nhãn):
  .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py list --diag-dir data/webcam_pilot

  # 3. [Xác nhận nhãn thật] Sau khi sửa nhãn sai bằng `set --diag-dir data/webcam_pilot --updates ...` (nếu có):
  .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py confirm-all --diag-dir data/webcam_pilot

  # 4. [Kiểm toán gần trùng] Xuất bảng metadata + cảnh báo ảnh gần trùng (không xóa ảnh, không tự đổi nhãn):
  .\.venv\Scripts\python.exe scripts/manage_webcam_samples.py list --diag-dir data/webcam_pilot --collection-audit results/svm/pilot_audit_v1

  # 5. [Chia tập pilot dry-run] Kiểm tra chia train/validation/test theo gesture_instance_id (0 rò rỉ lần giơ tay, chỉ kiểm tra kỹ thuật — hệ thống vẫn từ chối huấn luyện chính thức):
  .\.venv\Scripts\python.exe scripts/split_webcam_collection.py `
    --metadata results/svm/pilot_audit_v1/collection_metadata.csv `
    --pilot-instance-dry-run `
    --output results/svm/pilot_split_dry_run_v1
  ```

---

## 4. Kế hoạch thu đủ 36 lớp (`0–9, A–Z`) để huấn luyện SVM mới đáng tin cậy

Hiện tại trong `results/svm/webcam_diagnostics/` chỉ có **55 ảnh cũ** (chỉ 4 lớp `A: 11, B: 11, C: 18, D: 15`; thiếu **32/36 lớp**; và **55/55 ảnh thiếu 6 trường metadata**), nên hệ thống **chủ động từ chối huấn luyện** để không tạo kết quả giả.

### Mức tối thiểu khuyến nghị cho đợt dữ liệu 36 lớp đầu tiên
- **Số người ký hiệu (`participant_id`)**: tối thiểu **5 người** (`P01..P05`) — bắt buộc tối thiểu **3 người** để chia tách `train / validation / test` theo người (ví dụ: `Train: P01, P02, P03`, `Validation: P04`, `Test: P05`).
- **Mỗi người × mỗi ký hiệu (36 lớp)**: tối thiểu **5 lần giơ tay tách biệt** (`gesture_instance_id` khác nhau, mỗi lần hạ tay rồi giơ lại, thay đổi nhẹ góc/khoảng cách/nền/ánh sáng).
- **Tổng quy mô đợt 1**: `5 người × 36 lớp × 5 lần giơ = 900 ảnh` (tối thiểu kỹ thuật ở mức 3 người × 36 lớp × 3 lần giơ = `324 ảnh` để chạy đủ 3 tập, nhưng 5+ người và ≥ 5 lần giơ/người/lớp mới đủ ý nghĩa thống kê).
- **Đa dạng hóa có ghi metadata**:
  - `handedness`: ghi đúng `right` hoặc `left`.
  - `background`: ít nhất 2 loại nền (VD: `plain_wall`, `room_cluttered`).
  - `lighting`: ít nhất 2 điều kiện sáng (VD: `daylight`, `indoor_lamp`).

---

## 5. Lệnh chia tập & huấn luyện SVM (chỉ chạy sau khi đã đủ dữ liệu & xác nhận nhãn)

### 5.1. Chia `train / validation / test` tách theo người (cấm rò rỉ lần giơ tay)
```powershell
.\.venv\Scripts\python.exe scripts/split_webcam_collection.py `
  --metadata results/svm/collection_audit_full/collection_metadata.csv `
  --auto-assign `
  --output results/svm/webcam_split_v1
```
*(Hoặc truyền `--participants participants_split.json` để tự chỉ định người nào vào `train`/`validation`/`test`.)*

### 5.2. Huấn luyện phương án chính: HOG 1764 chiều + `LinearSVC` (nhất quán 100% với webcam)
```powershell
.\.venv\Scripts\python.exe scripts/train_webcam_svm.py `
  --split-csv results/svm/webcam_split_v1/split.csv `
  --samples-dir data/webcam_collection `
  --preprocess-modes legacy method_3b `
  --c-grid 0.01 0.1 1.0 10.0 `
  --refit-train-val
```
- Chọn `(preprocess_mode, C)` **hoàn toàn trên `validation`**.
- Chỉ đánh giá **1 lần** trên `test` độc lập sau khi chốt.
- Tự động so sánh với model hiện tại `models/svm/svm_final_c0_01.joblib` (cả Cách cũ và Cách 3B) trên **cùng tập test độc lập**.
- Lưu model mới riêng biệt tại `models/svm/webcam_hog_linear_svc_<timestamp>.joblib`.

### 5.3. Thí nghiệm tách biệt (tùy chọn): MediaPipe 21 Landmarks (42 chiều) + SVM
```powershell
.\.venv\Scripts\python.exe scripts/train_landmark_svm_experiment.py `
  --split-csv results/svm/webcam_split_v1/split.csv `
  --samples-dir data/webcam_collection `
  --kernels linear rbf `
  --c-grid 0.1 1.0 10.0
```
- Lưu riêng tại `models/svm/experiments/svm_landmarks_<timestamp>.joblib`.
- Dự đoán thử bằng `src/model/predict_landmark_svm.py` (dùng đúng hàm trích xuất 42 chiều lúc học, tự từ chối nếu truyền nhầm model HOG 1764 chiều).
