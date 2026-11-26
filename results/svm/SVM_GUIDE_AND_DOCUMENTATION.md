# Tài liệu Hướng dẫn Sử dụng và Tổng kết Thực nghiệm Mô hình SVM (ASL 36 Lớp)

Tài liệu này tổng hợp toàn bộ quy trình, kiến thức cốt lõi, kết quả thực nghiệm thực tế, các giới hạn dữ liệu đã kiểm chứng và hướng dẫn chạy dự đoán ảnh mới trên **Windows PowerShell** cho phần mô hình **SVM (`LinearSVC`)** trong đồ án Machine Learning.

---

## 1. Hướng dẫn Cài đặt và Chạy trên Windows PowerShell

Mọi lệnh dưới đây đều được chạy từ **thư mục gốc của dự án**:
```powershell
cd "C:\Users\Quang Diep\Machine-Learning---Sign-Language-Recognition"
```

### 1.1. Kiểm tra hoặc cài đặt thư viện vào môi trường ảo `.venv`
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
*(Phiên bản thư viện thực tế đã chạy: Python `3.13.6`, `scikit-learn 1.9.1`, `scikit-image 0.26.0`, `opencv-python 5.1.0`, `numpy 2.5.5`, `scipy 1.18.1`, `joblib 1.6.0`, `matplotlib 3.11.2`).*

### 1.2. Các lệnh đã thực sự chạy thành công trong dự án

1. **Xác minh baseline ban đầu (split ngẫu nhiên theo ảnh)**:
   ```powershell
   .\.venv\Scripts\python.exe src/model/verify_svm_baseline.py
   ```
2. **Tạo và kiểm chứng manifest 28.800 ảnh train + chia split theo nhóm tiền tố tên file (`P1..P10`)**:
   ```powershell
   .\.venv\Scripts\python.exe src/model/build_prefix_group_split.py
   ```
3. **Huấn luyện baseline `C = 1.0` trên split nhóm tiền tố**:
   ```powershell
   .\.venv\Scripts\python.exe src/model/train_svm.py --split-indices results/svm/svm_prefix_group_split_indices.npz --model-output models/svm/svm_baseline_c1_prefix_group.joblib --report-output results/svm/svm_baseline_c1_prefix_group_validation.json --c 1.0
   ```
4. **Chạy thực nghiệm 4 giá trị `C = 0.01, 0.1, 1.0, 10.0` trên split nhóm tiền tố**:
   ```powershell
   .\.venv\Scripts\python.exe scripts/run_svm_experiments.py
   ```
5. **Huấn luyện mô hình cuối cùng `LinearSVC(C=0.01)` trên 28.800 mẫu train và đánh giá trên 7.200 mẫu test**:
   ```powershell
   .\.venv\Scripts\python.exe src/model/train_final_svm.py
   ```
6. **Chạy dự đoán kiểm tra kỹ thuật trên một ảnh có đường dẫn chứa dấu cách và lưu kết quả JSON**:
   ```powershell
   .\.venv\Scripts\python.exe src/model/predict_svm.py "C:\Users\Quang Diep\Downloads\ASL-HG American Sign Language Hand Gesture Image D\ASL-HG American Sign Language Hand Gesture Image D\ASL_HG_36000\ASL_Processed_Images\asl_processed\test\0\P10_0_906.jpg" --json-output results/svm/svm_sample_prediction_P10_0_906.json
   ```

### 1.3. Lệnh minh họa để bạn tự thử một ảnh mới của mình

- **Hiển thị kết quả chi tiết trên màn hình PowerShell (hỗ trợ đường dẫn có dấu cách và tiếng Việt)**:
  ```powershell
  .\.venv\Scripts\python.exe src/model/predict_svm.py "C:\Đường dẫn của bạn\Ảnh thử nghiệm\ky_hieu_A.jpg"
  ```
- **Xuất thẳng kết quả dạng JSON ra màn hình hoặc lưu vào file JSON**:
  ```powershell
  .\.venv\Scripts\python.exe src/model/predict_svm.py "C:\Đường dẫn của bạn\Ảnh thử nghiệm\ky_hieu_A.jpg" --json
  .\.venv\Scripts\python.exe src/model/predict_svm.py "C:\Đường dẫn của bạn\Ảnh thử nghiệm\ky_hieu_A.jpg" --json-output "results/svm/ket_qua_anh_moi.json"
  ```
- **Tái sử dụng đối tượng dự đoán trong code Python (nạp model 1 lần duy nhất)**:
  ```python
  from src.model.predict_svm import SVMSignPredictor

  predictor = SVMSignPredictor("models/svm/svm_final_c0_01.joblib")
  result = predictor.predict_image("duong_dan_anh.jpg", top_k=3)
  print("Ký hiệu dự đoán:", result["predicted_label"])
  print("Top 3 decision scores:", result["top_k_predictions"])
  ```

### 1.4. Giải thích ý nghĩa các trường trong kết quả JSON
Khi chạy với `--json` hoặc `--json-output`, file JSON (ví dụ [svm_sample_prediction_P10_0_906.json](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_sample_prediction_P10_0_906.json)) gồm các trường chính:
- `predicted_label`: Tên ký hiệu dự đoán (ví dụ `"0"`, `"A"`, ..., `"Z"`), được ánh xạ trực tiếp từ `model.classes_` sang `class_names`.
- `predicted_class_index`: Chỉ số lớp nguyên (`0..35`) trong `model.classes_`.
- `predicted_decision_score`: Giá trị hàm quyết định tuyến tính (`decision_function`) của lớp đứng đầu. **Lưu ý**: Đây là khoảng cách đại số có dấu tới siêu phẳng phân lớp của `LinearSVC` (có thể âm hoặc dương), **không phải xác suất** và không phải độ tin cậy phần trăm.
- `margin_top1_minus_top2`: Hiệu số giữa `decision_score` của lớp đứng thứ 1 và lớp đứng thứ 2.
- `top_k_predictions`: Danh sách 3 lớp có `decision_score` cao nhất (`rank`, `class_index`, `label`, `decision_score`).
- `timing`:
  - `preprocess_and_hog_ms`: Thời gian (mili-giây) đọc file ảnh từ ổ đĩa, thu nhỏ giữ tỉ lệ + đệm nền trắng `64×64`, chuyển ảnh xám và trích xuất 1764 đặc trưng HOG.
  - `svm_predict_ms`: Thời gian (mili-giây) `LinearSVC` tính `decision_function` và chọn nhãn từ vector HOG 1764 chiều đã có trong RAM.
  - `total_end_to_end_ms`: Tổng thời gian xử lý toàn bộ 1 ảnh từ lúc đọc file đến khi ra nhãn.

---

## 2. Giải thích Kiến thức Cốt lõi (Gắn với Project)

### 2.1. Luồng xử lý: “Ảnh → Tiền xử lý → HOG → SVM → Ký hiệu”
1. **Đọc ảnh & Resize giữ tỉ lệ (`resize_with_padding` 64×64)**:
   - Ảnh gốc được đọc ở hệ màu BGR (`uint8`). Ta tính tỉ lệ thu phóng `scale = min(64 / rộng, 64 / cao)` để thu nhỏ bàn tay vừa khít khung mà **không làm méo tỉ lệ ngón tay**, rồi đặt vào chính giữa tấm nền trắng kích thước `64 × 64` (giá trị điểm ảnh `255`).
2. **Chuyển ảnh xám (Grayscale)**:
   - Biến ảnh màu `64 × 64 × 3` thành ma trận độ sáng `64 × 64` (`uint8`, từ `0` đến `255`), vì hình dáng ký hiệu phụ thuộc vào đường viền bàn tay chứ không phụ thuộc màu sắc.
3. **Trích xuất đặc trưng HOG (Histogram of Oriented Gradients — 1764 đặc trưng)**:
   - Chia ảnh `64 × 64` thành mạng lưới `8 × 8` ô nhỏ (`pixels_per_cell = (8, 8)`). Tại mỗi ô, tính độ nghiêng và độ mạnh của các đường viền (gradient) rồi gom vào `9` hướng góc (`orientations = 9`).
   - Gom mỗi cụm `2 × 2` ô liền kề thành một khối (`cells_per_block = (2, 2)`) và chuẩn hóa độ tương phản cục bộ theo chuẩn `L2-Hys`.
   - Có `7 × 7 = 49` vị trí khối trên ảnh `64 × 64`; mỗi khối có `2 × 2 × 9 = 36` con số $\Rightarrow$ tạo thành đúng **`49 × 36 = 1.764` đặc trưng `float32`** mô tả hình dáng bàn tay.
4. **Phân loại bằng `LinearSVC` (Support Vector Machine tuyến tính)**:
   - Với 36 lớp (`0–9, A–Z`), `LinearSVC` học **36 siêu phẳng phân cách** theo chiến lược *One-vs-Rest (OvR)*: mỗi lớp $k \in \{0..35\}$ có một vector trọng số $w_k \in \mathbb{R}^{1764}$ và độ lệch $b_k \in \mathbb{R}$.
   - Khi đưa vector HOG $x \in \mathbb{R}^{1764}$ vào, mô hình tính 36 điểm `decision_score`:
     $$s_k(x) = w_k^\top x + b_k$$
   - Lớp có $s_k(x)$ lớn nhất được chọn làm nhãn dự đoán.

### 2.2. Vai trò của Siêu tham số `C`
- `C` điều khiển mức đánh đổi giữa **độ rộng lề phân cách (margin / regularization)** và **mức phạt khi phân loại sai trên tập học**:
  - **`C` nhỏ (`0.01`)**: Ưu tiên giữ các trọng số $w_k$ nhỏ (regularization mạnh, lề rộng hơn), giúp mô hình ít bị quá khớp (overfit) vào những chi tiết riêng lẻ của các nhóm tiền tố trong tập `fit`.
  - **`C` lớn (`1.0, 10.0`)**: Phạt nặng mọi sai số trên tập học, ép siêu phẳng bám sát từng điểm dữ liệu học.

### 2.3. Vai trò của Train (Fit) / Validation / Test và các Chỉ số Đánh giá
- **Fit (`23.026` mẫu, 8 nhóm tiền tố `P1..P7, P9`)**: Tập dữ liệu để `pipeline.fit()` học trọng số $w_k, b_k$.
- **Validation (`5.774` mẫu, 2 nhóm tiền tố `P8, P10`)**: Tập “thi thử” tách biệt hoàn toàn theo nhóm tiền tố tên file, chỉ dùng để so sánh và chọn `C = 0.01`.
- **Test (`7.200` mẫu)**: Tập đánh giá cuối cùng của cả nhóm sau khi đã chốt `C = 0.01` và huấn luyện lại trên toàn bộ `28.800` mẫu train.
- **Các chỉ số**:
  - **Accuracy**: Tỉ lệ dự đoán đúng trên tổng số mẫu.
  - **Precision (của lớp $k$)**: Trong số các ảnh mô hình đoán là lớp $k$, bao nhiêu phần trăm thực sự là lớp $k$.
  - **Recall (của lớp $k$)**: Trong số tất cả ảnh thật của lớp $k$, mô hình tìm đúng được bao nhiêu phần trăm.
  - **F1-score**: Trung bình điều hòa giữa Precision và Recall ($2 \cdot \frac{P \cdot R}{P + R}$); **Macro F1** là trung bình cộng F1 của cả 36 lớp (coi mọi lớp quan trọng ngang nhau).
  - **Confusion Matrix**: Bảng $36 \times 36$ cho biết mỗi lớp thật (hàng) được đoán đúng bao nhiêu mẫu (đường chéo chính) và bị nhầm sang lớp nào (các ô ngoài đường chéo).

---

## 3. Tổng hợp Kết quả Thực nghiệm Thực tế và Các Giới hạn Quan trọng

### 3.1. Bảng tổng hợp kết quả qua các giai đoạn

| Giai đoạn / Thực nghiệm | Cách chia & Tập đánh giá | Fit Accuracy | Val / Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Thời gian Fit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ban đầu (`C=1.0`)** | Split ngẫu nhiên theo ảnh (`23.040` fit / `5.760` val) | `100.00%` | `100.00%` (`5760/5760`) | `1.000000` | `1.000000` | `1.000000` | `130.318s` |
| **Thực nghiệm `C=0.01` *(Được chọn)*** | Split tách nhóm tiền tố (`Fit: P1..P7,P9` / `Val: P8,P10`) | `100.00%` | **`65.4659%`** (`3780/5774`) | **`0.722645`** | `0.653407` | **`0.636364`** | `96.714s` |
| **Thực nghiệm `C=0.1`** | Split tách nhóm tiền tố (`Fit: P1..P7,P9` / `Val: P8,P10`) | `100.00%` | `65.5178%` (`3783/5774`) | `0.718496` | `0.654078` | `0.635587` | `91.855s` |
| **Thực nghiệm `C=1.0` *(Baseline nhóm)*** | Split tách nhóm tiền tố (`Fit: P1..P7,P9` / `Val: P8,P10`) | `100.00%` | `65.2234%` (`3766/5774`) | `0.697273` | `0.651103` | `0.633226` | `73.334s` |
| **Thực nghiệm `C=10.0`** | Split tách nhóm tiền tố (`Fit: P1..P7,P9` / `Val: P8,P10`) | `100.00%` | `65.3793%` (`3775/5774`) | `0.701511` | `0.652454` | `0.635365` | `99.454s` |
| **Mô hình cuối cùng (`C=0.01`)** | Train `28.800` mẫu → Đánh giá trên **Test gốc (`7.200` mẫu)** | `99.9931%` (`28798/28800`) | **`99.9861%`** (`7199/7200`) | `0.999862` | `0.999861` | `0.999861` | `86.343s` |
| **Đối chứng khử trùng byte (`C=0.01`)** | Test gốc sau khi trừ 165 ảnh trùng SHA-256 (`7.035` mẫu) | `99.9931%` | `99.9858%` (`7034/7035`) | `0.999860` | `0.999832` | `0.999846` | *(Dùng chung model)* |

### 3.2. Các giới hạn và lưu ý bắt buộc khi báo cáo kết quả
1. **Không khẳng định `P1–P10` là 10 người thực hiện ký hiệu**:
   - Thư mục dữ liệu cục bộ và hai file `.zip` gốc không có file README hay bảng metadata xác minh `P1..P10`. Ngoài ra, kiểm chứng mã băm SHA-256 phát hiện 131 cặp file trùng byte hoàn toàn giữa `P5` và `P6` ở lớp `I` và `V`. Vì vậy, chỉ gọi `P1..P10` là **“nhóm tiền tố tên file”**.
2. **Không mô tả `99.9861%` là độ chính xác trên người mới hoặc ảnh thực tế bất kỳ**:
   - Tập `test` gốc (`7.200` ảnh) tuy không có file trùng tên với `train` (`28.800` ảnh), nhưng **chia sẻ cả 10 nhóm tiền tố `P1..P10`** (`360/360` cụm `(P_k, lớp)`), có **`165` ảnh trùng khít mã băm SHA-256** với ảnh trong `train`, và có **`95.79%` mẫu test** mang số thứ tự tên file liền kề (`±1`) với ảnh trong `train`.
   - Việc loại 165 ảnh trùng byte (`99.9858%` trên `7.035` mẫu) **chưa loại bỏ được các ảnh gần giống nhau hoặc sự phụ thuộc giữa các nhóm tiền tố** cùng xuất hiện ở cả `train` và `test`.
   - Sự khác biệt giữa điểm Validation tách nhóm tiền tố (`~65.47%`) và điểm Test gốc (`99.9861%`) là do cấu trúc chia dữ liệu của hai tập khác nhau, **không phải mức cải thiện do mô hình**.
3. **Ghi nhận minh bạch về quy trình đánh giá Giai đoạn B**:
   - Trong Giai đoạn B, mặc dù bước kiểm tra trước huấn luyện phát hiện `165` ảnh test trùng byte SHA-256 với train và sự trùng lặp toàn bộ 10 nhóm tiền tố `P1..P10`, quá trình chạy đã tiếp tục đánh giá trên tập `test` gốc thay vì dừng lại theo điều kiện dừng của yêu cầu Giai đoạn B. Toàn bộ kết quả và file minh chứng được giữ nguyên không che giấu, đồng thời hạn chế của tập `test` gốc được công bố đầy đủ như trên.
4. **Nguyên nhân mẫu sai duy nhất `P8_I_746.jpg` (`I → X`) là giả thuyết**:
   - Trên tập `test`, chỉ có 1 mẫu bị đoán sai là `P8_I_746.jpg` (nhãn thật `I`, dự đoán `X`). Việc giải thích rằng cả `I` và `X` đều có hình nắm tay với một ngón nhô lên nên dễ có phân bố gradient HOG gần nhau ở góc chụp nghiêng là một **giả thuyết trực quan**, chưa có bằng chứng thực nghiệm can thiệp trực tiếp trên ảnh đó.

---

## 4. Điều kiện Ảnh Đầu vào khi Thử Ảnh Mới và Hướng Phát triển

1. **Cách chuẩn bị ảnh đầu vào để phù hợp với pipeline hiện tại**:
   - Dữ liệu huấn luyện trong `asl_processed` là ảnh **bàn tay đã được cắt sát khung (cropped hand region)** trên **nền sạch/đồng nhất**.
   - Nếu bạn tự chụp ảnh mới để thử với `src/model/predict_svm.py`, cần **cắt (crop) khung hình bao sát bàn tay**, để bàn tay nằm rõ ở giữa khung với nền đơn giản, ánh sáng đều và góc chụp chính diện tương tự dữ liệu học.
2. **Giới hạn khi đưa ảnh ngoài đời vào pipeline hiện tại**:
   - Pipeline hiện tại **không có bước tự động phát hiện bàn tay (hand detection / MediaPipe crop) hay tách nền**. Nếu đưa ảnh chụp toàn thân hoặc ảnh có hậu cảnh nhiều đồ vật, thuật toán HOG sẽ trích xuất cả các đường viền của hậu cảnh, làm sai lệch vector 1764 chiều.
   - Mô hình luôn chọn lớp có `decision_score` cao nhất trong 36 lớp (`0–9, A–Z`), nên ảnh không có bàn tay hoặc ký hiệu lạ vẫn sẽ bị gán vào 1 trong 36 nhãn.
3. **Đề xuất bước tiếp theo**:
   - Thu thập một tập ảnh kiểm thử nhỏ hoàn toàn độc lập từ bên ngoài bộ dữ liệu (được cắt khung bàn tay rõ ràng) để đánh giá thực tế khả năng tổng quát hóa của mô hình `svm_final_c0_01.joblib`.

---

## 5. Hướng dẫn Chạy Nhận diện Camera Trực tiếp trên Giao diện SignVision (Windows PowerShell)

### 5.1. Kiến trúc kết nối Camera → SVM → Giao diện
1. **Backend Python ([svm_api_server.py](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/src/model/svm_api_server.py))**:
   - Sử dụng `ThreadingHTTPServer` chuẩn của Python để tương thích trực tiếp với `.venv` hiện có mà không cần cài thêm framework ngoài.
   - Nạp model [svm_final_c0_01.joblib](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/models/svm/svm_final_c0_01.joblib) **một lần duy nhất** khi khởi động thông qua `SVMSignPredictor`.
   - Giải mã ảnh base64 JPEG ngay trong bộ nhớ RAM (`cv2.imdecode`), thực hiện đúng pipeline `resize_with_padding` `64×64` → grayscale → HOG 1764 chiều (`float32`) → `LinearSVC(C=0.01)` mà **không ghi khung hình camera ra ổ đĩa**.
2. **Frontend SignVision ([RecognitionPage.tsx](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/frontend/src/recognition/RecognitionPage.tsx), [useHandLandmarks.ts](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/frontend/src/recognition/useHandLandmarks.ts), [usePredictor.ts](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/frontend/src/recognition/usePredictor.ts))**:
   - **Phát hiện và cắt vùng bàn tay (ROI)**:
     - **Chế độ mặc định (`MediaPipe HandLandmarker`)**: Phát hiện bàn tay trên khung hình gốc. Nếu có nhiều tay, chọn **1 bàn tay có diện tích hộp giới hạn lớn nhất** (`selectLargestHand`), cắt vùng bàn tay kèm khoảng đệm `28%` (`computePaddedHandCropBox`) giới hạn chặt trong khung ảnh để giữ đủ các đầu ngón tay. Khi không thấy tay, hiển thị **“Chưa phát hiện bàn tay”**, xóa kết quả cũ và không gửi request.
     - **Chế độ dự phòng (`Khung hướng dẫn cố định`)**: Nếu môi trường không tải được MediaPipe hoặc người dùng chọn trong phần *Cài đặt nhận diện*, giao diện hiển thị ô vuông hướng dẫn cố định ở giữa màn hình để người dùng đặt tay vào (ghi rõ chế độ này không phát hiện có tay hay không và luôn phân loại vào 1 trong 36 lớp).
   - **Phân biệt lật gương (Mirror)**: Video xem trước có thể bật hiệu ứng lật gương để thuận mắt người dùng, nhưng ảnh vùng bàn tay cắt gửi về model SVM luôn lấy trực tiếp từ khung hình gốc **không lật gương**.
   - **Chống dồn hàng đợi & Làm mượt hiển thị**:
     - Tốc độ gửi mặc định **4 khung hình/giây** (cấu hình được từ `1–10` FPS).
     - Tối đa **1 yêu cầu dự đoán đang xử lý** tại mỗi thời điểm (`busy` lock); tự động bỏ qua khung hình mới khi request trước chưa xong và hủy kết quả trễ sau khi tắt camera hoặc mất bàn tay.
     - Bỏ phiếu đa số trên **3 kết quả gần nhất** (`applyMajorityVote`) để giảm nhấp nháy nhãn hiển thị, tự động đặt lại khi mất tay hoặc tắt camera.

### 5.2. Lệnh khởi động Backend và Frontend (Mở 2 cửa sổ PowerShell)

**Cửa sổ PowerShell 1 — Khởi động Backend Python SVM API (cổng 8000):**
```powershell
cd "C:\Users\Quang Diep\Machine-Learning---Sign-Language-Recognition"
.\.venv\Scripts\python.exe src/model/svm_api_server.py --host 127.0.0.1 --port 8000
```
*(Kiểm tra nhanh trạng thái máy chủ tại `http://localhost:8000/health`).*

**Cửa sổ PowerShell 2 — Khởi động Giao diện Frontend SignVision (cổng 5173):**
```powershell
cd "C:\Users\Quang Diep\Machine-Learning---Sign-Language-Recognition\frontend"
npm run dev
```

### 5.3. Các bước thao tác kiểm tra trên trình duyệt
1. Mở trình duyệt tại địa chỉ Vite hiển thị (thường là `http://localhost:5173`). Giao diện mở sẵn ở màn hình **Nhận diện** (hoặc nhấn mục **Nhận diện** trên thanh điều hướng).
2. Kiểm tra dòng trạng thái bên dưới khung camera hiển thị chấm xanh: **`SVM API sẵn sàng (svm_final_c0_01.joblib)`**.
3. Nhấn nút **“Bật camera”** (hoặc **“Cho phép truy cập camera”**) và chấp nhận quyền sử dụng webcam trên trình duyệt.
4. Giơ **một bàn tay** lên trước camera ở nơi đủ sáng, nền phía sau đơn giản:
   - Ở chế độ **MediaPipe** (mặc định): Khung viền xanh lá sẽ bao quanh bàn tay lớn nhất và gửi vùng cắt đó về SVM. Khi bỏ tay ra khỏi khung hình, trạng thái chuyển thành **“Chưa phát hiện bàn tay”** và xóa dự đoán cũ.
   - Ở chế độ **Khung hướng dẫn cố định** (chuyển trong *Cài đặt nhận diện (SVM)* nếu muốn): Đặt bàn tay lọt vào ô vuông nét đứt ở chính giữa màn hình.
5. Quan sát thẻ **Kết quả hiện tại (SVM)**: hiển thị ký hiệu dự đoán (`0–9, A–Z`), **`Decision score (#1)`**, độ chênh lệch `(#1 − #2)` và **Top 3 lớp theo `decision_score`** (không phải xác suất phần trăm).
6. Nhấn **“Tắt camera”** hoặc chuyển sang trang khác để dừng vòng lặp và giải phóng webcam.

