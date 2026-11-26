# Báo cáo Tổng kết Hướng A: Nhận diện Ngôn ngữ Ký hiệu ASL 36 lớp bằng MediaPipe 21 Hand Landmarks + SVM

**Dự án:** SignVision — Machine Learning Sign Language Recognition  
**Phạm vi:** 36 ký hiệu ASL tĩnh (`0–9`, `A–Z`)  
**Ngày chốt báo cáo:** 2026-11-26  
**Nguyên tắc:** Toàn bộ số liệu trong báo cáo được trích nguyên văn từ các file kết quả JSON/CSV hiện có trong thư mục `results/svm/experiments/`.

---

## 1. Bối cảnh & Phương pháp

Trước Hướng A, hệ thống sử dụng pipeline **HOG + LinearSVC** (`models/svm/svm_final_c0_01.joblib`, cấu hình **HOG Cách 3B**: ảnh cắt bàn tay `64×64` grayscale, đặc trưng HOG `1.764` chiều). Dù hoạt động ổn định trên tập ảnh cắt chuẩn, HOG rất nhạy với nền phòng, ánh sáng và góc xoay khi đưa ra webcam thật.

**Hướng A** chuyển sang trích xuất **21 điểm mốc bàn tay 2D từ MediaPipe Hand Landmarker (`mediapipe 1.2.3`)** kết hợp mô hình phân loại **SVM (`StandardScaler + SVC(kernel='rbf')` / `LinearSVC`)** — lưu ý điểm khác biệt rõ ràng về thuật toán so với hướng dùng `RandomForestClassifier` trong nhiều video tham khảo phổ biến:
- **Mô hình được giữ nguyên ở lõi thuật toán SVM** theo định hướng môn học/dự án, sử dụng nhân `RBF` (`SVC(kernel='rbf')`) để xử lý biên phi tuyến giữa các cấu hình khớp ngón tay.
- **Tương thích 100% giữa huấn luyện và suy luận trực tuyến:** hàm chuẩn hóa đặc trưng duy nhất trong `src/preprocess/webcam_features.py` đọc trực tiếp 21 tọa độ chuẩn hóa `(x, y)` cùng kích thước khung hình `(image_width, image_height)`, khớp hoàn toàn với `HandFrame.hands` từ `@mediapipe/tasks-vision` ở frontend.
- **Bảo toàn mốc đối chiếu (Baseline):** toàn bộ 7 file model HOG cũ trong `models/svm/` (bao gồm `models/svm/svm_final_c0_01.joblib`), endpoint `POST /predict` và các báo cáo cũ được giữ nguyên ở backend để đối chiếu công bằng.

---

## 2. Kiểm soát rò rỉ dữ liệu trên `asl_processed` (Đợt 1)

Kiểm toán bộ dữ liệu `asl_processed` (`36.000` ảnh = `36` lớp × `1.000` ảnh/lớp) phát hiện rò rỉ nghiêm trọng ở phép chia `train / test` gốc:

| Hạng mục kiểm toán | Kết quả đo thực tế | Nguồn JSON |
| :--- | :--- | :--- |
| Tập gốc (`train` / `test`) | `28.800` ảnh (`28.391` SHA-256 duy nhất) / `7.200` ảnh (`7.172` SHA-256 duy nhất) | [asl_leakage_audit.json](./landmark_asl_v1/asl_leakage_audit.json) |
| Rò rỉ `prefix-group` ở chia gốc | **`10 / 10`** nhóm (`P1..P10`) xuất hiện xen kẽ ở cả `train` và `test` gốc | [asl_leakage_audit.json](./landmark_asl_v1/asl_leakage_audit.json) |
| Trùng byte SHA-256 giữa `train` & `test` gốc | **`159`** mã SHA-256 duy nhất trùng (**`165`** dòng ở `test` gốc có bản sao hệt ở `train` gốc; `227` dòng ngược lại) | [asl_leakage_audit.json](./landmark_asl_v1/asl_leakage_audit.json) |
| Trùng SHA-256 liên nhóm trên `36.000` ảnh | **`199`** cụm trùng hoàn toàn nằm ở cặp **`P5–P6`** | [asl_leakage_audit.json](./landmark_asl_v1/asl_leakage_audit.json) |
| Khả năng dùng `test` gốc đánh giá tách người | **`False`** (`can_use_original_test_as_unseen_group_test = false`) | [asl_leakage_audit.json](./landmark_asl_v1/asl_leakage_audit.json) |

### Phép chia chống rò rỉ theo thành phần liên thông (`connected_component_prefix_group_split`)
Gộp cứng cặp `{P5, P6}` thành một super-group `P5+P6` đưa vào `train`, chia tách hoàn toàn 3 tập:

| Tập dữ liệu | Nhóm `prefix-group` | Tổng ảnh | MediaPipe bắt được tay | Tỷ lệ phát hiện | Số mẫu tối thiểu/lớp |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`train`** | `P1, P3, P4, P5, P6, P9` | `21.600` | `20.803` | `96.31%` | `403` (lớp `0`) |
| **`validation`** | `P8, P10` | `7.200` | `6.932` | `96.28%` | `76` |
| **`test`** | `P2, P7` | `7.200` | `6.742` | `93.64%` | `94` |
| **Tổng cộng** | `P1..P10` | **`36.000`** | **`34.477`** | **`95.77%`** | — |

- **Xác minh rò rỉ sau chia lại:** `prefix_group_overlap_count = 0`, `sha256_overlap_count = 0` ([split_summary.json](./landmark_asl_v1/split_summary.json)).
- **Các lớp có tỷ lệ phát hiện MediaPipe thấp nhất** (`border_pad_ratio = 0.5`, `min_detection_confidence = 0.3`): `0` (`57.3%`), `P` (`67.5%`), `Q` (`86.1%`), `O` (`86.4%`), `G` (`87.4%`) ([detection_summary_by_class.json](./landmark_asl_v1/detection_summary_by_class.json)).
- **Lệch tay trái/phải nghiêm trọng & cách khắc phục:** Trong `34.477` ảnh bắt được tay trên `asl_processed`, có **`34.365` tay phải (`99.68%`)** và chỉ **`112` tay trái (`0.32%`)**. Khi huấn luyện không lật ngang (`aug0`), ứng viên tốt nhất (`linear_C0.1_aug0`) đạt `82.92%` trên `validation` gốc nhưng sụp xuống **`2.51%`** trên `validation` lật ngang (mô phỏng tay đối diện). Do đó, mọi mô hình từ `v1` đến `v3_split` đều áp dụng **tăng cường lật ngang (`horizontal flip`) chỉ trên tập `train`** (tuyệt đối không chạm `validation`/`test`), giúp cân bằng hoàn toàn hiệu năng hai tay (`val_orig` ≈ `val_mirrored`).

---

## 3. Tiến hóa mô hình trên `asl_processed` (`v1` → `v2` → `v3_split`)

Toàn bộ việc chọn mô hình và siêu tham số chỉ thực hiện trên tập **`validation (P8, P10)`**, sau đó đánh giá cuối **1 lần** trên tập **`test (P2, P7)`**:

### 3.1. Mô hình 36 lớp: `v1` (42 chiều) vs `v2` (86 chiều hình học)

- **`v1` (`mediapipe_21_xy_wrist_norm_aspect_v2`, 42 chiều):** chuẩn hóa 21 điểm `(x, y)` gốc tại cổ tay (landmark `0`), giữ tỷ lệ khung hình, chia theo bán kính cực đại. Trên `validation`, `SVC(rbf)` (`rbf_C10_gscale_aug1`) vượt trội `LinearSVC` tốt nhất (`linear_C0.1_aug1`: `71.71%` Cond Acc / `0.6950` Macro-F1).
- **`v2` (`mediapipe_21_xy_geom_v2`, 86 chiều):** giữ nguyên 42 tọa độ gốc và bổ sung 44 đặc trưng hình học bất biến xoay tính trực tiếp từ 21 điểm `(x, y)` (khoảng cách đầu ngón–cổ tay, đầu ngón–khớp MCP, khoảng cách cặp đầu ngón, góc cosin các khớp PIP/DIP/MCP và tọa độ xoay thẳng theo trục cổ tay→MCP giữa `0→9`), xuất phát từ chẩn đoán hình học `C/D/0/1` ([phase2b_cd_geometry_diagnosis.json](./landmark_asl_v2/phase2b_cd_geometry_diagnosis.json)).

| Phiên bản | Đặc trưng & cấu hình chọn trên `validation` | Val (`P8, P10`) Cond Acc / Macro-F1 / E2E | Test (`P2, P7`) Cond Acc / Macro-F1 / E2E | Test Mirrored Cond Acc | Suy luận (ms/mẫu) | Nguồn |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`v1` (36 lớp)** | 42D · `SVC(rbf, C=10, gamma='scale') + flip` | `80.05%` / `0.7922` / `77.07%` | **`84.32%`** / **`0.8273`** / **`78.96%`** | `84.35%` | `1.29` | [phase2_val](./landmark_asl_v1/phase2_validation_comparison.json) · [phase2_test](./landmark_asl_v1/phase2_test_evaluation.json) |
| **`v2` (36 lớp)** | 86D · `SVC(rbf, C=1.0, gamma=0.005) + flip` | **`88.21%`** / **`0.8715`** / **`84.93%`** | `83.48%` / `0.8205` / `78.17%` | `83.48%` | `6.97` | [phase2b_val](./landmark_asl_v2/phase2b_validation_comparison.json) · [phase2b_test](./landmark_asl_v2/phase2b_test_evaluation.json) |

> **Nhận xét `v1` vs `v2`:** `v2` tăng **`+8.16%`** Cond Acc trên `validation` (`80.05%` → `88.21%`). Trên `test (P2, P7)` giảm nhẹ `-0.85%` (`84.32%` → `83.48%`), nhưng trên **webcam thật** (`32` mẫu độc lập và `55` mẫu cũ), `v2` cải thiện vượt bậc **`+15.62%` đến `+16.37%` E2E** so với `v1` nhờ giảm phụ thuộc vào góc nghiêng cổ tay.

### 3.2. Tách phạm vi Chữ cái `A–Z` & Chữ số `0–9` (`v3_split`, 86 chiều)

Thử nghiệm trực tiếp `v2` trên webcam thật ghi nhận 4 nhóm lỗi nổi bật: **`I ↔ J`**, **`K → 2`**, **`V → 2`**, **`X → 1`** (3/4 lỗi là va chạm chéo giữa Chữ cái và Chữ số; ngay trên `test P2, P7`, `v2` 36 lớp dự đoán lọt ngoài tập con `404` mẫu ở `A–Z` — riêng `V → 2` chiếm `101` mẫu — và `57` mẫu ở `0–9`).

Đợt 3 so sánh có kiểm soát giữa giữ model 36 lớp (`v2_36_full`), che nhãn theo phạm vi (`v2_36_masked`), và huấn luyện 2 mô hình chuyên biệt `SVM_AZ` (26 lớp) & `SVM_09` (10 lớp) cùng trên 86D + `flip` train:

| Tập con | Phương pháp | Val (`P8, P10`) Cond Acc / Macro-F1 / E2E | Test (`P2, P7`) Cond Acc / Macro-F1 / E2E | Lỗi lọt ngoài tập con (Test) | Suy luận (ms/mẫu) | Nguồn |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Chữ cái `A–Z`** (26 lớp) | `v2_36_full` (36 lớp gốc) | `88.29%` / `0.8763` / `85.85%` | `78.42%` / `0.8007` / `73.88%` | `404` | `6.97` | [phase3_val](./landmark_asl_v3_split/phase3_validation_comparison.json) |
| **Chữ cái `A–Z`** (26 lớp) | `v2_36_masked` (che nhãn) | `89.38%` / `0.8815` / `86.90%` | `86.69%` / `0.8688` / `81.67%` | `0` | `6.97` | [phase3_test](./landmark_asl_v3_split/phase3_test_evaluation.json) |
| **Chữ cái `A–Z`** (26 lớp) | **`SVM_AZ` (`C=10, g=0.005, flip`)** | **`91.38%`** / **`0.8993`** / **`88.85%`** | **`87.88%`** / **`0.8813`** / **`82.79%`** | **`0`** | **`1.16`** | [phase3_test](./landmark_asl_v3_split/phase3_test_evaluation.json) |
| **Chữ số `0–9`** (10 lớp) | `v2_36_full` (36 lớp gốc) | `88.01%` / `0.9131` / `82.55%` | `96.91%` / `0.9639` / `89.30%` | `57` | `6.97` | [phase3_val](./landmark_asl_v3_split/phase3_validation_comparison.json) |
| **Chữ số `0–9`** (10 lớp) | `v2_36_masked` (che nhãn) | `98.83%` / `0.9779` / `92.70%` | **`99.89%`** / **`0.9986`** / **`92.05%`** | `0` | `6.97` | [phase3_test](./landmark_asl_v3_split/phase3_test_evaluation.json) |
| **Chữ số `0–9`** (10 lớp) | **`SVM_09` (`C=10, g=0.01, flip`)** | **`99.41%`** / **`0.9899`** / **`93.25%`** | `99.84%` / `0.9974` / `92.00%` | **`0`** | **`0.12`** | [phase3_test](./landmark_asl_v3_split/phase3_test_evaluation.json) |

- Trên `test (P2, P7)`, `SVM_AZ` triệt tiêu `V → 2` từ `101` mẫu về **`0`** (recall `V` tăng `42.29%` → `93.14%`), cải thiện `X` từ `65.75%` → `81.77%`, đồng thời nhanh hơn ~`6×` (`1.16 ms/mẫu`) so với `v2_36_full` ([phase3_four_errors_diagnosis.json](./landmark_asl_v3_split/phase3_four_errors_diagnosis.json)).

---

## 4. Đối chiếu định lượng trên Webcam thật

### 4.1. Các tập kiểm thử 4 lớp `A/B/C/D` (đã xác nhận nhãn)

| Tập ảnh webcam (4 lớp `A/B/C/D`) | Mẫu (`det/total`) | HOG Cách 3B (E2E / Cond) | Landmark `v1` 42D (E2E / Cond) | Landmark `v2` `all_36` (E2E / Cond) | Landmark `SVM_AZ` (E2E / Cond) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `data/webcam_pilot` (`P01/S01`, tay trái) | `14 / 15` (`93.33%`) | `80.00%` / `78.57%` | `73.33%` / `78.57%` | `73.33%` / `78.57%` | **`86.67%`** / **`92.86%`** |
| `webcam_diagnostics` — `32` mẫu độc lập | `31 / 32` (`96.88%`) | `53.12%` / `51.61%` | `37.50%` / `38.71%` | `53.12%` / `54.84%` | **`62.50%`** / **`64.52%`** |
| `webcam_diagnostics` — `23` mẫu cũ (tập chọn 3B) | `23 / 23` (`100.0%`) | `78.26%` / `78.26%` | `60.87%` / `60.87%` | `78.26%` / `78.26%` | **`82.61%`** / **`82.61%`** |
| `webcam_diagnostics` — toàn bộ `55` mẫu cũ | `54 / 55` (`98.18%`) | `63.64%` / `62.96%` | `47.27%` / `48.15%` | `63.64%` / `64.81%` | **`70.91%`** / **`72.22%`** |

*Nguồn:* [phase2_webcam_benchmark.json](./landmark_asl_v1/phase2_webcam_benchmark.json), [phase2b_webcam_3way_benchmark.json](./landmark_asl_v2/phase2b_webcam_3way_benchmark.json), [phase3_webcam_4way_benchmark.json](./landmark_asl_v3_split/phase3_webcam_4way_benchmark.json). Các mẫu chưa xác nhận nhãn (`label_confirmed = False`) luôn bị loại khỏi tính điểm.

### 4.2. Tập kiểm thử webcam độc lập đủ 36 lớp (`data/webcam_eval_36`, `77` mẫu đã xác nhận nhãn)

- **Kiểm kê kỹ thuật ([collection_audit.json](./webcam_eval_36_v1/audit_confirmed/collection_audit.json)):** `77 / 77` mẫu đã đối chiếu bằng mắt và xác nhận nhãn (`label_confirmed = True`), phủ đủ **`36 / 36` lớp** (`2–4` lần giơ độc lập/lớp, `77` lần giơ), `1` người chụp (`P01 / S01`), `0` trùng SHA-256, `1` mẫu chạm mép (`#71`, lớp `2`), `4` cặp gần trùng `MAE ≤ 8`, tỷ lệ phát hiện MediaPipe **`76 / 77` (`98.70%`)** (chỉ `#57` lớp `6` không bắt được tay).

| Phạm vi đánh giá | Mẫu (`det/total`) | **HOG Cách 3B** (E2E / Cond / Macro-F1) | **Landmark `all_36`** (E2E / Cond / Macro-F1) | **Landmark chuyên biệt `A–Z` / `0–9`** (E2E / Cond / Macro-F1) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng 36 lớp (`0–9, A–Z`)** | `76 / 77` (`98.70%`) | `25/77` = `32.47%` / `32.89%` / `0.2954` | `47/77` = `61.04%` / `61.84%` / `0.5827` | **`49/77` = `63.64%` / `64.47%` / `0.5958`** |
| **Chữ cái `A–Z` (26 lớp)** | `52 / 52` (`100.00%`) | `16/52` = `30.77%` / `30.77%` / `0.2876` | `32/52` = `61.54%` / `61.54%` / `0.6231` | **`33/52` = `63.46%` / `63.46%` / `0.5987`** |
| **Chữ số `0–9` (10 lớp)** | `24 / 25` (`96.00%`) | `9/25` = `36.00%` / `37.50%` / `0.4105` | `15/25` = `60.00%` / `62.50%` / `0.5690` | **`16/25` = `64.00%` / `66.67%` / `0.5882`** |

- **Số lớp đạt 100% đúng trên 36 lớp:** HOG Cách 3B chỉ đạt **`6 / 36`** (`3, 9, B, G, L, T`); Landmark `all_36` đạt **`18 / 36`**; Landmark chuyên biệt (`A–Z` / `0–9`) đạt **`19 / 36`** (`0, 3, 4, 5, 9, A, B, C, D, E, F, H, I, L, O, P, V, W, Y`).
- **Tác động rõ rệt của tách phạm vi trên webcam thật:**
  - Chặn dứt điểm các va chạm chữ–số `0 → B` (`0`: `2/4` → **`4/4`**), `D → 1` (`D`: `1/2` → **`2/2`**), `V → 2` (`V`: `1/2` → **`2/2`**), và đẩy lỗi `K → 2`, `X → 1` ra khỏi biên chéo chữ–số.
  - Các lỗi tồn đọng trên bản chuyên biệt là cặp hình học gần nhau trong cùng phạm vi: `G → H (2)`, `N → Q (2)`, `X → Z (2)`, `S → M (2)`, `J → I (2)`, `7 → 8 (2)`, `1 → 0 (2)`.
- **Báo cáo chi tiết:** [webcam_eval_36_report.json](./webcam_eval_36_v1/webcam_eval_36_report.json) · [webcam_eval_36_per_class.csv](./webcam_eval_36_v1/webcam_eval_36_per_class.csv) · [webcam_eval_36_samples.csv](./webcam_eval_36_v1/webcam_eval_36_samples.csv) · [webcam_eval_36_errors.csv](./webcam_eval_36_v1/webcam_eval_36_errors.csv).

---

## 5. Kiến trúc ứng dụng SignVision hiện tại

- **Frontend (`frontend/`, React 19 + TypeScript + Vite 8, cổng dev `8443`):**
  - MediaPipe Hand Landmarker (`@mediapipe/tasks-vision 1.2.3`) chạy trực tiếp trên trình duyệt để theo dõi bàn tay và gửi 21 tọa độ `(x, y)` cùng kích thước khung hình tới backend (`POST /predict/landmark`) với tần suất mặc định `4 Hz`.
  - **Chế độ mặc định trên UI là Landmark SVM** với bộ chọn gọn **3 phạm vi:** `Tất cả 36` (`all_36`), `Chữ cái A–Z` (`letters_az`), `Chữ số 0–9` (`digits_09`).
  - Làm mượt theo cửa sổ đa số (`smoothingWindow = 3`) và chốt ký tự vào câu sau `stability = 4` khung hình ổn định. Toàn bộ điều khiển HOG Cách 3B đã được ẩn khỏi UI để tối giản trải nghiệm.
- **Backend (`src/model/svm_api_server.py`, Python `ThreadingHTTPServer`, mặc định `127.0.0.1:8000`):**
  - Tải 1 lần khi khởi động cả 3 bundle Landmark SVM 86D qua `LandmarkV2ScopeRegistry` (`src/model/predict_landmark_v2.py`):
    - `all_36`: `models/svm/experiments/landmark_asl_v2/svm_landmark_asl_v2.joblib`
    - `letters_az`: `models/svm/experiments/landmark_asl_v3_split/svm_landmark_az_v3.joblib`
    - `digits_09`: `models/svm/experiments/landmark_asl_v3_split/svm_landmark_09_v3.joblib`
  - **Độ trễ HTTP thấp:** chỉ truyền JSON 21 điểm mốc thay vì ảnh nén base64; thời gian suy luận SVM nội bộ ở mức `~0.1–7 ms/mẫu`.
  - **Hiệu chuẩn `confidence` không bão hòa:** tính ước lượng đơn điệu trong `(0, 1)` từ số phiếu OvO và chênh lệch biên cộng dồn khôi phục từ `decision_function` OvR của `SVC(kernel='rbf')` (`sigmoid(0.8 * vote_gap + 3.5 * delta_avg - 0.5)`), đã kiểm chứng `0.00%` bão hòa `> 0.99` trên `validation (P8, P10)` ở cả 3 phạm vi mà không làm thay đổi nhãn dự đoán hay `decision_score` gốc.
  - **Baseline HOG Cách 3B giữ nguyên:** `models/svm/svm_final_c0_01.joblib`, endpoint `POST /predict`, `GET /reference-samples`, và `POST /diagnostic/save` vẫn hoạt động đầy đủ ở backend.

---

## 6. Giới hạn trung thực & Hướng mở rộng

1. **Giới hạn quy mô người chụp trên webcam thật:** tập đánh giá độc lập 36 lớp `data/webcam_eval_36` hiện chỉ có **`77` mẫu từ `1` người chụp (`P01 / S01`)** (`2–4` mẫu/lớp, có `1` mẫu chạm mép và `4` cặp gần trùng `MAE ≤ 8`). Các con số **`63.64%` E2E (`64.47%` Cond)** phản ánh thực tế trên người chụp hiện tại và **chưa đủ để khẳng định độ tổng quát nhiều người** (cần thu thêm tối thiểu **`≥ 3` người chụp độc lập** dưới nhiều điều kiện ánh sáng/góc camera).
2. **Hai ký hiệu động `J` và `Z`:** trong ASL chuẩn, `J` và `Z` được xác định bởi quỹ đạo chuyển động của ngón tay. Với bộ phân loại ảnh tĩnh 1 khung hình, `J` tĩnh trùng dáng `I` (`J → I` ở `2/2` mẫu webcam) và `Z` chỉ đạt `0/2`. Cần bổ sung bộ đệm chuỗi ngắn (ví dụ `10–20` khung hình liên tiếp) trích xuất vector dịch chuyển đầu ngón út (`landmark 20` cho `J`) và ngón trỏ (`landmark 8` cho `Z`).
3. **Domain shift tư thế tay ở các lớp yếu (`6, 7, 8, G, M, N, S, X`):** dù đã vượt xa HOG Cách 3B (`32.47%` → `63.64%`), vẫn tồn tại khoảng cách rõ giữa `test (P2, P7)` (`87.88%` trên `A–Z`, `99.84%` trên `0–9`) và webcam thật (`63.46%` trên `A–Z`, `66.67%` Cond trên `0–9`) do khác biệt góc bàn tay 3D khi người dùng giơ trước camera laptop so với ảnh tĩnh của `asl_processed` (và mất chiều sâu `z`). Hướng cải thiện thực tế tiếp theo là thu thập tập hiệu chỉnh đa người chụp (tách nhóm chặt theo `participant_id`) và thử nghiệm bổ sung đặc trưng chiều sâu tương đối `z` chuẩn hóa từ MediaPipe 3D.
