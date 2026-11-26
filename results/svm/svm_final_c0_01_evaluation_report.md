# Báo cáo Huấn luyện và Đánh giá Cuối cùng — Mô hình SVM (`LinearSVC C=0.01`)

## 1. Kiểm tra trước khi huấn luyện và Nguồn gốc tập dữ liệu

1. **Cấu hình đã chốt từ Giai đoạn A**:
   - Mô hình: `Pipeline([("classifier", LinearSVC(C=0.01, random_state=42, max_iter=10000, dual="auto"))])`, không dùng scaler.
   - Siêu tham số `C = 0.01` được chọn hoàn toàn dựa trên **Validation Macro F1 chưa làm tròn cao nhất (`0.636364`)** trên tập validation chia theo nhóm tiền tố (`P8, P10`) trong Giai đoạn A, hoàn toàn chưa chạm vào tập `test`.
2. **Kiểm tra kích thước, nhãn lớp và mã băm dữ liệu (`data/processed/svm_hog_features.npz`)**:
   - `X_train`: `(28800, 1764)`, `float32`, hữu hạn 100% (`np.isfinite = True`).
   - `y_train`: `(28800,)`, `int32`, đủ 36 lớp (`0..9, A..Z`), đúng **800 mẫu/lớp**.
   - `X_test`: `(7200, 1764)`, `float32`, hữu hạn 100% (`np.isfinite = True`).
   - `y_test`: `(7200,)`, `int32`, đủ 36 lớp (`0..9, A..Z`), đúng **200 mẫu/lớp**.
   - Mã băm SHA-256 của file đặc trưng và các mảng khớp 100% với biên bản Giai đoạn A (`096667c46efa4a3d160043fc1c5c32ce9febf1c910d179a2a0191f036db16402`).
3. **Kiểm chứng ánh xạ Manifest tập Test và quan hệ giữa `train` và `test` gốc**:
   - Đã ánh xạ toàn bộ 7.200 đường dẫn ảnh trong `asl_processed/test` và lưu tại [svm_test_image_manifest.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_test_image_manifest.csv).
   - Nhãn thư mục khớp **7.200 / 7.200** với `y_test`; trích xuất lại HOG ở ảnh đầu và ảnh cuối của toàn bộ **360 / 360 khối `(P_k, lớp)`** trong tập test (720 ảnh biên) khớp tuyệt đối từng bit (`720/720`) với `X_test`.
   - **Những gì xác minh được về trùng nhóm và trùng dữ liệu giữa `train` (28.800) và `test` (7.200)**:
     - **Trùng tên file**: **`0`** (không có file nào trùng tên giữa `train/` và `test/`).
     - **Trùng nhóm tiền tố tên file (`P1..P10`)**: Cả **10 / 10** nhóm tiền tố `P1..P10` và cả **360 / 360** cụm `(P_k, lớp)` đều xuất hiện ở cả `train` (80 ảnh/cụm) và `test` (20 ảnh/cụm) theo kiểu xen kẽ số thứ tự tên file.
     - **Số thứ tự liền kề (`số_thứ_tự ± 1`)**: Có **6.897 / 7.200** mẫu test (`95.79%`) có ảnh mang số thứ tự liền kề ngay trước hoặc ngay sau trong cùng cụm `(P_k, lớp)` nằm ở tập `train`.
     - **Trùng nội dung byte file ảnh (SHA-256) và vector HOG**: Có **165 / 7.200** mẫu test (`2.29%`) có mã băm SHA-256 và vector HOG trùng khít với ít nhất một ảnh khác tên trong tập `train` (ví dụ: ảnh test `P10_0_906.jpg` trùng byte với ảnh train `P10_0_905.jpg`).
     - **Kết luận về tính chất tập Test và ghi nhận quy trình**:
       - Tập `test` gốc của bộ dữ liệu **không phải là tập độc lập theo nhóm tiền tố**, cũng không chứng minh được khả năng nhận diện trên người mới hay ảnh thực tế bất kỳ.
       - **Ghi nhận minh bạch về điều kiện dừng ở Giai đoạn B**: Trong yêu cầu Giai đoạn B có nêu điều kiện *"Nếu gặp vấn đề nghiêm trọng về dữ liệu hoặc rò rỉ train/test, hãy dừng trước khi đánh giá test"*. Tuy nhiên, ở lượt chạy Giai đoạn B trước đó, script `src/model/train_final_svm.py` đã tiếp tục chạy đánh giá trên tập `test` ngay sau khi phát hiện 165 ảnh trùng byte và sự chia sẻ toàn bộ 10 nhóm tiền tố `P1..P10` giữa `train` và `test`. Để đảm bảo tính trung thực khoa học, toàn bộ kết quả và file đã tạo được **giữ nguyên, không xóa hay che giấu**, nhưng phải diễn giải đúng giới hạn:
         1. Con số **`99.9861%` (`7.199 / 7.200`) không được mô tả là độ chính xác trên người mới hoặc ảnh thực tế bất kỳ**.
         2. Việc tính thêm chỉ số sau khi loại 165 ảnh trùng byte (`7.034 / 7.035` đúng, `99.9858%`) **chỉ loại bỏ các bản sao trùng khít từng byte, chứ chưa loại bỏ được các ảnh gần giống nhau** (95.79% mẫu test có số thứ tự liền kề `±1` trong train) hoặc sự phụ thuộc do dùng chung các nhóm tiền tố `P1..P10`.
         3. Không khẳng định `P1..P10` là 10 người thực hiện ký hiệu vì không có tài liệu metadata chính thức đi kèm bộ dữ liệu cục bộ xác minh điều này.

---

## 2. Kết quả Huấn luyện và Đánh giá Cuối cùng

### 2.1. Tổng hợp kết quả theo từng giai đoạn (Phân biệt rõ mục đích từng tập)

> [!IMPORTANT]
> **Không gọi sự chênh lệch giữa Validation theo nhóm tiền tố (`65.47%`) và Test gốc (`99.99%`) là mức cải thiện do mô hình.** Đây là hai tập đánh giá có cấu trúc phân phối khác hẳn nhau:
> - **Validation Giai đoạn A (`5.774` mẫu thuộc `P8, P10`)**: Đánh giá trên **2 nhóm tiền tố bị tách biệt hoàn toàn khỏi tập `fit` (`P1..P7, P9`)**, dùng để chọn siêu tham số `C = 0.01`.
> - **Test Giai đoạn B (`7.200` mẫu thuộc `P1..P10`)**: Đánh giá mô hình cuối cùng (đã học trên cả 10 nhóm tiền tố `P1..P10` của tập `train` 28.800 mẫu) trên tập `test` dùng chung của nhóm (vốn cũng chứa `P1..P10`).

| Tập đánh giá / Mục đích | Số mẫu | Đúng / Sai | Accuracy (Chưa làm tròn) | Macro Precision | Macro Recall | Macro F1 (Chưa làm tròn) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Validation nhóm tiền tố (Giai đoạn A — Chọn `C=0.01`)**<br>*(Fit: `23.026` mẫu `P1..P7,P9` → Val: `P8,P10`)* | `5.774` | `3.780` / `1.994` | `0.654659` (`65.47%`) | `0.722645` (`72.26%`) | `0.653407` (`65.34%`) | `0.636364` (`63.64%`) |
| **2. Toàn bộ tập Train (Giai đoạn B — Kiểm tra độ khớp)**<br>*(Huấn luyện trên toàn bộ `28.800` mẫu `P1..P10`)* | `28.800` | `28.798` / `2` | `0.999931` (`99.9931%`) | `0.999931` (`99.9931%`) | `0.999931` (`99.9931%`) | `0.999931` (`99.9931%`) |
| **3. Tập Test chuẩn của nhóm (Giai đoạn B — Đánh giá cuối)**<br>*(Đánh giá 1 lần trên toàn bộ `7.200` mẫu `X_test`)* | **`7.200`** | **`7.199` / `1`** | **`0.999861`** (**`99.9861%`**) | **`0.999862`** (**`99.9862%`**) | **`0.999861`** (**`99.9861%`**) | **`0.999861`** (**`99.9861%`**) |
| **4. Tập Test đã khử 165 ảnh trùng byte SHA-256**<br>*(Đối chứng minh bạch khi bỏ 165 mẫu trùng byte với train)* | `7.035` | `7.034` / `1` | `0.999858` (`99.9858%`) | `0.999860` (`99.9860%`) | `0.999832` (`99.9832%`) | `0.999846` (`99.9846%`) |

### 2.2. Thông số hội tụ và Thời gian chạy thực tế
- **Hội tụ khi huấn luyện trên 28.800 mẫu**: Hội tụ hoàn toàn (`converged = True`), số vòng lặp tối ưu `n_iter_max = 13` (trung vị `13.0`), **`0` cảnh báo `ConvergenceWarning`**.
- **Thời gian huấn luyện (`28.800` mẫu × `1.764` đặc trưng)**: **`86.3428` giây**.
- **Thời gian dự đoán trên toàn bộ tập Test (`7.200` mẫu)**:
  - Tổng thời gian cho `7.200` mẫu: **`0.0460` giây** (`46.05` ms).
  - Trung bình mỗi mẫu: **`0.0064` ms/mẫu** (`~6.4` micro-giây/mẫu).
  - *(Lưu ý phạm vi đo)*: Thời gian này **chỉ đo bước dự đoán của `LinearSVC` từ vector đặc trưng HOG 1764 chiều đã trích xuất sẵn trong bộ nhớ RAM**, chưa bao gồm thời gian đọc file ảnh từ ổ đĩa, resize/padding về `64×64` và trích xuất HOG.

---

## 3. Kết quả theo từng lớp (36 lớp) và Phân tích nhầm lẫn (Confusion Matrix)

- **34 / 36 lớp** (gồm toàn bộ 10 chữ số `0..9` và 24 chữ cái `A..H, J..W, Y, Z`) đạt **`Precision = 1.000000`, `Recall = 1.000000`, `F1 = 1.000000`** (đoán đúng tuyệt đối `200 / 200` mẫu mỗi lớp).
- Chỉ có **2 / 36 lớp** có sai số do **duy nhất 1 mẫu bị nhầm lẫn** trên toàn bộ 7.200 mẫu test:

| Lớp | Support | Đoán đúng | Đoán sai | Precision | Recall | F1-score | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`I`** | `200` | `199` | `1` | `1.000000` | `0.995000` | `0.997494` | 1 mẫu thật là `I` bị đoán nhầm sang `X` |
| **`X`** | `200` | `200` | `0` | `0.995025` | `1.000000` | `0.997506` | Đoán đúng cả 200 mẫu `X`, nhận nhầm 1 mẫu từ `I` |
| **34 lớp còn lại** | `200`/lớp | `200`/lớp | `0`/lớp | `1.000000` | `1.000000` | `1.000000` | Tuyệt đối `200/200` mẫu mỗi lớp |

### Các cặp nhầm lẫn phổ biến nhất (Nhãn thật → Nhãn dự đoán)
Trên toàn bộ `7.200` mẫu test, chỉ có **1 cặp nhầm lẫn duy nhất** khác 0:
1. **`I → X`**: **`1` mẫu** (`0.50%` của lớp `I`), cụ thể là file ảnh `P8_I_746.jpg` (`row_index = 3774`, nhóm tiền tố `P8`, `decision_score = -0.340309`, khoảng cách biên `top1 - top2 = 0.200368`, không phải ảnh trùng SHA-256 với train).
   - **Giả thuyết về nguyên nhân nhầm lẫn (Hypothesis)**: Trong bảng ký hiệu ASL, ký hiệu **`I`** (nắm bàn tay, giơ ngón út) và ký hiệu **`X`** (nắm bàn tay, giơ ngón trỏ uốn cong) đều có phần nắm tay phía dưới và một ngón tay nhô lên phía trên. Một **giả thuyết hợp lý** là khi thu nhỏ về ảnh xám `64×64` và tính biểu đồ hướng cạnh HOG, góc chụp hoặc độ nghiêng của ngón tay trong ảnh `P8_I_746.jpg` tạo ra phân bố gradient gần với vùng biên của lớp `X`. *(Lưu ý: Đây là giả thuyết giải thích trực quan dựa trên đặc trưng hình học của HOG, chưa phải kết luận đã được chứng minh trực tiếp bằng thực nghiệm can thiệp ảnh)*.

---

## 4. Danh sách file minh chứng đã lưu và Kiểm chứng toàn vẹn

- **Model cuối cùng**: [svm_final_c0_01.joblib](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/models/svm/svm_final_c0_01.joblib) (đã kiểm chứng nạp lại bằng `joblib.load()`, dự đoán khớp 100% với `7.200` mẫu test).
- **Báo cáo JSON đầy đủ**: [svm_final_c0_01_test_evaluation.json](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_test_evaluation.json)
- **Bảng chỉ số 36 lớp (CSV)**: [svm_final_c0_01_per_class_metrics.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_per_class_metrics.csv)
- **Danh sách các cặp nhầm lẫn (CSV)**: [svm_final_c0_01_top_confusions.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_top_confusions.csv)
- **Nhãn thật & nhãn dự đoán 7.200 mẫu test**: [svm_final_c0_01_test_predictions.npz](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_test_predictions.npz) và [svm_final_c0_01_test_predictions.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_test_predictions.csv)
- **Confusion Matrix (Tổng = 7.200 khớp tuyệt đối kích thước test)**:
  - Dạng số lượng: [svm_final_c0_01_confusion_matrix_counts.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_confusion_matrix_counts.csv) và [svm_final_c0_01_confusion_matrix_counts.png](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_confusion_matrix_counts.png)
  - Dạng chuẩn hóa theo nhãn thật: [svm_final_c0_01_confusion_matrix_normalized.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_confusion_matrix_normalized.csv) và [svm_final_c0_01_confusion_matrix_normalized.png](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_final_c0_01_confusion_matrix_normalized.png)
- **Manifest 7.200 ảnh test**: [svm_test_image_manifest.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_test_image_manifest.csv)
