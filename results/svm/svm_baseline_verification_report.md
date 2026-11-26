# Báo cáo xác minh Baseline SVM (LinearSVC C=1.0 — Validation 100%)

## 1. Đã kiểm tra gì

1. **Kiểm tra code huấn luyện (`src/model/train_svm.py`) và đối chiếu dự đoán từ model đã lưu (`models/svm/svm_baseline_c1.joblib`)**:
   - Kiểm tra xem `pipeline.fit()` có chỉ dùng phần `X_fit` (80% của `X_train`), `pipeline.predict()` có chỉ dùng `X_validation` (20% của `X_train`), và các hàm tính điểm có so sánh đúng `y_validation` với `y_pred` mà không gán cứng hay làm tròn che giấu lỗi hay không.
   - Tải trực tiếp model đã lưu (không train lại) để đếm chính xác số mẫu dự đoán đúng/sai và biên quyết định (`decision_function`).
2. **Tái tạo tập chia (split) và kiểm tra giao nhau về chỉ số dòng (`index`)**:
   - Tái lập phép chia `train_test_split(..., test_size=0.20, random_state=42, stratify=y_train)` kèm mảng chỉ số `0..28799`.
   - Lưu chỉ số `fit_indices` và `validation_indices` cùng mã băm SHA-256 (fingerprint) của dữ liệu vào `results/svm/svm_validation_split_indices.npz`.
3. **Kiểm tra trùng lặp vector đặc trưng HOG và trùng lặp file ảnh gốc**:
   - Đối chiếu từng dòng `float32` (1764 chiều) giữa `X_validation` (5.760 mẫu) và `X_fit` (23.040 mẫu).
   - Đối chiếu mã băm SHA-256 của từng file ảnh `.jpg` gốc trên ổ đĩa để phân biệt **trùng vector HOG** và **trùng file ảnh**.
   - Tính khoảng cách L2 và độ tương đồng cosine của láng giềng gần nhất (1-NN) từ mỗi mẫu validation sang tập `X_fit`.
4. **Kiểm tra metadata, ánh xạ đường dẫn ảnh và quy luật đặt tên file**:
   - Kiểm tra toàn bộ thư mục dataset và 2 file `.zip` gốc xem có file metadata/README xác nhận mã người ký hiệu hay không.
   - Ánh xạ lại danh sách 28.800 đường dẫn ảnh train và 7.200 đường dẫn ảnh test theo đúng thứ tự duyệt thư mục của `src/preprocess/build_hog_dataset.py`, đồng thời kiểm chứng chéo bằng nhãn `y_train`, `y_test` và trích xuất lại HOG trên 144 ảnh mẫu trải đều 36 lớp.
   - Phân tích cấu trúc tên file `{P1..P10}_{Lớp}_{Số_thứ_tự}.jpg` và mức độ xen kẽ giữa `fit`, `validation` và `test`.

---

## 2. Bằng chứng và số liệu thực tế

### 2.1. Kiểm tra code và dự đoán của model đã lưu
- Trong `src/model/train_svm.py`:
  - Dòng 82–95: `load_training_data()` chỉ nạp `X_train`, `y_train`, `class_names`; hoàn toàn không nạp `X_test`, `y_test`.
  - Dòng 178: `pipeline.fit(X_fit, y_fit)` chỉ học trên 23.040 mẫu `fit`.
  - Dòng 182–193: `y_pred = pipeline.predict(X_validation)` dự đoán trên 5.760 mẫu `validation` và so sánh trực tiếp `y_validation` với `y_pred` bằng `accuracy_score`, `precision_recall_fscore_support`. Không có chỉ số nào bị gán cứng hoặc làm tròn che lỗi.
- Kết quả chạy lại `pipeline.predict()` từ `models/svm/svm_baseline_c1.joblib`:
  - **Tập Fit (23.040 mẫu)**: Dự đoán đúng **23.040 / 23.040** mẫu, sai **0** mẫu (`accuracy = 1.0`).
  - **Tập Validation (5.760 mẫu)**: Dự đoán đúng **5.760 / 5.760** mẫu, sai **0** mẫu (`accuracy = 1.0`, `macro_precision = 1.0`, `macro_recall = 1.0`, `macro_f1 = 1.0`).
  - **Biên quyết định (`decision_function` top-1 trừ top-2 trên validation)**: nhỏ nhất `0.9764`, trung vị `2.2835`, lớn nhất `3.4642` (tức mẫu sát ranh giới nhất vẫn cách lớp đứng thứ hai gần `0.98` đơn vị).

### 2.2. Kiểm tra tập chỉ số chia (Split Indices) & Fingerprint dữ liệu
- Số lượng chỉ số: `len(fit_indices) = 23040` (640 mẫu/lớp × 36 lớp), `len(validation_indices) = 5760` (160 mẫu/lớp × 36 lớp).
- Số chỉ số giao nhau giữa `fit` và `validation`: **0** (`index_overlap_count = 0`), hợp của hai tập khớp đúng 28.800 dòng (`0..28799`).
- Mảng `X_train[fit_indices]` và `X_train[validation_indices]` khớp tuyệt đối từng bit (`np.array_equal = True`) với `X_fit`, `X_validation` trong `train_svm.py`.
- Đã lưu file tái lập split tại `results/svm/svm_validation_split_indices.npz` kèm fingerprint SHA-256:
  - `svm_hog_features.npz` (`163,463,525` bytes): `096667c46efa4a3d160043fc1c5c32ce9febf1c910d179a2a0191f036db16402`
  - `X_train`: `76cfd91db0907e1640174a938ec343e23b1b2a68b941fee555bf374f3d144ec9`
  - `y_train`: `c12cfcd6ca3c7f93f830d1d88c20898fa7f0268e2b8dcf10e12f7a7332d16dc8`

### 2.3. Kiểm tra trùng lặp vector HOG và trùng file ảnh gốc
- **Trùng hoàn toàn vector HOG giữa `validation` và `fit`**: Có **128 / 5.760** mẫu validation (`2.22%`) có vector HOG trùng khít 100% với ít nhất một mẫu trong tập `fit`.
- **Trùng hoàn toàn file ảnh gốc (SHA-256 của file `.jpg`) giữa `validation` và `fit`**: Cũng có đúng **128 / 5.760** mẫu validation trùng mã băm SHA-256 file `.jpg` với ít nhất một file ảnh khác tên trong tập `fit`.
  - Mọi trường hợp trùng vector HOG đều xuất phát từ các file ảnh `.jpg` có nội dung byte giống hệt nhau nhưng mang tên file khác nhau trong dataset gốc.
  - Ví dụ thực tế:
    - Mẫu validation `P2_9_158.jpg` (dòng `7410`) trùng 100% file và HOG với mẫu fit `P2_9_159.jpg` (dòng `7411`) và `P2_9_160.jpg` (dòng `7412`).
    - Mẫu validation `P8_H_792.jpg` (dòng `14313`) trùng 100% file và HOG với mẫu fit `P8_H_793.jpg` (dòng `14314`).
    - Mẫu validation `P5_I_481.jpg` (dòng `14864`) trùng 100% file và HOG với mẫu fit `P6_I_581.jpg` (dòng `14948`).
- **Độ tương đồng láng giềng gần nhất (1-NN) trên toàn bộ 5.760 mẫu validation**:
  - **5.759 / 5.760** mẫu validation (`99.98%`) có mẫu HOG gần nhất trong tập `fit` thuộc cùng nhãn lớp.
  - Độ tương đồng cosine với mẫu `fit` gần nhất có trung vị đạt **`0.9828`** (nhỏ nhất `0.8729`, lớn nhất `1.0000`).

### 2.4. Kiểm tra metadata và cấu trúc tên file ảnh
- **Metadata**: File `svm_hog_features.npz` không lưu đường dẫn ảnh. Kiểm tra thư mục dataset và hai file nén `ASL_Processed_Images.zip`, `ASL_Raw_Images.zip` xác nhận chỉ có 36.000 file ảnh `.jpg`, **không có file README hay bảng metadata nào** giải thích ý nghĩa của tiền tố `P1..P10`.
- **Kiểm chứng ánh xạ đường dẫn (`0..28799`)**:
  - Thứ tự nhãn tái tạo từ thư mục khớp 100% với `y_train` (`28.800` mẫu) và `y_test` (`7.200` mẫu).
  - Chạy lại `extract_hog()` trên **108 ảnh train** (đầu, giữa, cuối của cả 36 lớp) và **36 ảnh test** cho kết quả khớp tuyệt đối từng bit (`108/108` và `36/36`) với `X_train` và `X_test` trong file `.npz`.
- **Quy luật tên file và sự xen kẽ theo nhóm tiền tố tên file**:
  - 100% ảnh (28.800 ảnh train và 7.200 ảnh test) có định dạng `{P1..P10}_{Lớp}_{Số_thứ_tự}.jpg`, trong đó mỗi nhóm tiền tố tên file gắn với một dải 100 số thứ tự cố định cho mỗi lớp: `P1` (`1..100`), `P2` (`101..200`), ..., `P10` (`901..1000`).
  - Trong 100 ảnh của mỗi cụm `(P_k, Lớp)`, dataset gốc đã chia 80 ảnh vào `train/` và 20 ảnh vào `test/` theo kiểu xen kẽ số thứ tự trong tên file (ví dụ lớp `0`, nhóm tiền tố `P10`: `901..905` ở `train`, `906` ở `test`, `907..912` ở `train`, `913` ở `test`).
  - Khi chia ngẫu nhiên `train` thành `fit` (80%) và `validation` (20%):
    - Cả **360 / 360** cụm `(P_k, Lớp)` đều xuất hiện đồng thời ở cả 3 tập `fit`, `validation` và `test`.
    - **4.980 / 5.760** mẫu validation (`86.46%`) có ảnh mang số thứ tự liền kề trong tên file (`số_thứ_tự ± 1`, cùng nhóm tiền tố `P_k` và cùng lớp) nằm trong tập `fit`.
    - **5.703 / 5.760** mẫu validation (`99.01%`) có láng giềng HOG gần nhất trong `fit` nằm cùng cụm `(P_k, Lớp)`. Ví dụ: `P4_1_373.jpg` (validation) có láng giềng gần nhất là `P4_1_374.jpg` (fit) với `cosine_similarity = 0.9904`.

---

## 3. Kết luận được hỗ trợ bởi bằng chứng

1. **Không có lỗi lập trình trong code baseline**: `src/model/train_svm.py` tách dữ liệu đúng kỹ thuật, không dùng nhầm tập `fit` hay `test` khi đánh giá `validation`, không có giao nhau về chỉ số dòng (`overlap = 0`), và con số `5.760 / 5.760` dự đoán đúng là kết quả thật từ model `LinearSVC(C=1.0)`.
2. **Nguyên nhân đạt 100% trên validation đến từ việc chia ngẫu nhiên theo từng ảnh khi dữ liệu có độ tương đồng rất cao trong cùng nhóm tiền tố tên file**:
   - Thứ nhất, có **128 mẫu validation** (`2.22%`) trùng 100% nội dung byte (SHA-256) và vector HOG với ảnh khác tên trong tập `fit` (gồm 86 mẫu trùng trong cùng nhóm tiền tố `P_k` và 42 mẫu trùng xuyên nhóm giữa `P5` và `P6` ở hai lớp `I`, `V`).
   - Thứ hai (giải thích cho 97.78% mẫu còn lại), mỗi lớp gồm 10 nhóm tiền tố tên file (`P1..P10`), và phép chia ngẫu nhiên theo ảnh làm cho cả 10 nhóm tiền tố đều xuất hiện ở hai phía, trong đó **86.46%** mẫu validation có ảnh mang số thứ tự tên file liền kề (`số_thứ_tự ± 1`) nằm trong tập `fit`, dẫn tới vector HOG giữa `validation` và `fit` rất gần nhau (trung vị cosine similarity đạt `0.9828`).

---

## 4. Điều gì vẫn chưa biết (Giới hạn hiện tại)

- **Ý nghĩa chính thức của `P1..P10`**: Trong thư mục dataset đã tải và hai file `.zip` gốc hoàn toàn không có file README hay bảng metadata chính thức xác nhận `P1..P10` là gì. Hơn nữa, kiểm chứng thực nghiệm SHA-256 trên file cục bộ cho thấy giữa `P5` và `P6` có 131 cặp file trùng byte hoàn toàn ở lớp `I` và `V`. Vì vậy, để trung thực với bằng chứng thực tế, báo cáo chỉ gọi `P1..P10` là **“nhóm tiền tố tên file”** chứ không khẳng định đó là 10 người thực hiện ký hiệu hay chuỗi khung hình video liên tiếp.
- **Khả năng tổng quát hóa**: Vì cả `fit`, `validation` (trong split ngẫu nhiên) và `test` gốc đều chứa chung 10 nhóm tiền tố `P1..P10`, kết quả 100% trên validation ngẫu nhiên **không chứng minh** khả năng nhận diện trên người mới hay bối cảnh chụp mới.

---

## 5. Đề xuất bước tiếp theo

- **Giữ nguyên split và dữ liệu hiện tại** (không tự ý xóa 128 ảnh trùng hay sửa cấu trúc dataset gốc để đảm bảo tính nhất quán với đề bài và các thành viên khác trong nhóm), nhưng **ghi nhận rõ phát hiện này vào báo cáo đồ án** như một phân tích chất lượng dữ liệu sâu sắc.
- **Chuyển sang Giai đoạn A (Thử nghiệm siêu tham số `C = 0.01, 0.1, 1.0, 10.0`)**:
  - Sử dụng chính xác tập chỉ số đã lưu trong `results/svm/svm_validation_split_indices.npz`.
  - Ghi nhận đầy đủ accuracy trên `fit` và `validation`, macro precision/recall/F1 chưa làm tròn, thời gian huấn luyện và cảnh báo hội tụ.
  - Chọn `C` theo quy tắc đã định: ưu tiên `macro_f1` validation cao nhất (chưa làm tròn); nếu bằng nhau (ví dụ nhiều mức `C` cùng đạt `1.0`), chọn `C` nhỏ hơn để ưu tiên biên phân cách rộng hơn (regularization mạnh hơn).
