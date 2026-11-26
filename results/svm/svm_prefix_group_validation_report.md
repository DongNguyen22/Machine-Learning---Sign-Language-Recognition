# Báo cáo Đánh giá Baseline SVM (LinearSVC C=1.0) theo Nhóm Tiền Tố Tên File (`P1..P10`)

## 1. Mục tiêu và nguyên tắc thực hiện

1. **Giữ nguyên kết quả cũ làm đối chứng**:
   - Không xóa dữ liệu gốc; giữ nguyên model [svm_baseline_c1.joblib](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/models/svm/svm_baseline_c1.joblib), báo cáo [svm_baseline_c1_validation.json](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_baseline_c1_validation.json) và tập chỉ số chia ngẫu nhiên [svm_validation_split_indices.npz](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_validation_split_indices.npz).
2. **Xác minh thuật ngữ cho `P1..P10`**:
   - Kiểm tra toàn bộ thư mục dữ liệu cục bộ và hai file nén `ASL_Processed_Images.zip`, `ASL_Raw_Images.zip` xác nhận **không có file README hay bảng metadata chính thức đi kèm** để xác minh ý nghĩa của `P1..P10`.
   - Kiểm chứng mã băm SHA-256 trên dữ liệu thực tế còn cho thấy giữa `P5` và `P6` tồn tại **131 cặp file trùng khít từng byte** ở hai lớp `I` (67 cặp) và `V` (64 cặp).
   - Vì vậy, để bám sát bằng chứng thực nghiệm và không kết luận vượt quá dữ liệu, báo cáo chỉ gọi `P1..P10` là **“nhóm tiền tố tên file”** và gọi phương pháp đánh giá này là **“đánh giá theo nhóm tiền tố”**, không khẳng định `P1..P10` là 10 người thực hiện ký hiệu và không gọi đây là “đánh giá người mới”.

---

## 2. Xác minh ánh xạ đường dẫn ảnh (Manifest) và kiểm tra trùng lặp SHA-256

Trước khi chia nhóm bằng script [build_prefix_group_split.py](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/src/model/build_prefix_group_split.py):
- **Ánh xạ 28.800 dòng `X_train`**:
  - Nhãn thư mục tái tạo khớp **28.800 / 28.800** dòng với `y_train`.
  - Trích xuất lại HOG ở ảnh đầu tiên và ảnh cuối cùng của toàn bộ **360 / 360 khối `(P_k, lớp)`** (tổng cộng **720 / 720** ảnh biên) cho kết quả khớp tuyệt đối từng bit (`np.array_equal = True`) với `X_train`.
  - Đã lưu bảng ánh xạ đầy đủ (gồm `row_index`, `relative_path`, `filename`, `prefix_group`, `class_name`, `class_index`, `sequence_number`, `file_sha256`) tại [svm_train_image_manifest.csv](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_train_image_manifest.csv) (SHA-256 manifest: `e88339f4e26c4a4dc544bc67b512782f9d34c81bb181e081effde38b2dc0d43c`).
- **Quy tắc chia nhóm tiền tố cố định (trước khi xem điểm mô hình)**:
  - Dùng `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` trên cột `prefix_group` (`P1..P10`), giữ nguyên toàn bộ ảnh có cùng tiền tố `P_k` ở một phía (không chia theo `(P_k, lớp)`).
  - **Phía `fit` (8 nhóm tiền tố)**: `P1, P2, P3, P4, P5, P6, P7, P9` — tổng **23.026** mẫu (`79.95%`), phủ đủ **36 / 36** lớp (mỗi lớp từ `628` đến `657` mẫu).
  - **Phía `validation` (2 nhóm tiền tố)**: `P8, P10` — tổng **5.774** mẫu (`20.05%`), phủ đủ **36 / 36** lớp (mỗi lớp từ `143` đến `172` mẫu).
- **Kiểm tra SHA-256 và vector HOG xuyên nhóm**:
  - Giữa `fit` mới (`P1..P7, P9`) và `validation` mới (`P8, P10`): **0** ảnh trùng mã băm SHA-256 và **0** vector HOG trùng nhau.
  - Rà soát toàn bộ $\binom{10}{2} = 45$ cặp nhóm tiền tố trong 28.800 ảnh train: 44/45 cặp có **0** ảnh trùng; duy nhất cặp `('P5', 'P6')` có **131** cặp ảnh trùng SHA-256 (67 cặp lớp `I`, 64 cặp lớp `V`, ví dụ `P5_I_401.jpg` trùng byte với `P6_I_501.jpg`). Trong split hiện tại, cả `P5` và `P6` đều nằm trong tập `fit`, nên không gây rò rỉ byte giữa `fit` và `validation`. (Nếu về sau thực hiện chia chéo nhiều fold tách `P5` và `P6` sang hai phía, cần gộp `P5+P6` hoặc loại bỏ 131 bản sao trùng byte này trước khi đánh giá).

---

## 3. So sánh Split cũ (Ngẫu nhiên theo ảnh) và Split mới (Theo nhóm tiền tố)

| Tiêu chí | Split cũ (Ngẫu nhiên theo ảnh) | Split mới (Theo nhóm tiền tố `P_k`) |
| :--- | :--- | :--- |
| Cách chia | `train_test_split(stratify=y_train, test_size=0.2, random_state=42)` | `GroupShuffleSplit(groups=prefix_group, test_size=0.2, random_state=42)` |
| Nhóm tiền tố ở `fit` | Cả 10 nhóm `P1..P10` | 8 nhóm: `P1, P2, P3, P4, P5, P6, P7, P9` |
| Nhóm tiền tố ở `validation` | Cả 10 nhóm `P1..P10` | 2 nhóm: `P8, P10` |
| Giao nhau nhóm tiền tố | **10 / 10** nhóm xuất hiện ở cả hai phía | **0** nhóm giao nhau |
| Số mẫu `fit` / `validation` | `23.040` / `5.760` (đủ 36 lớp) | `23.026` / `5.774` (đủ 36 lớp) |
| Trùng file SHA-256 (Val vs Fit) | **128** mẫu (`2.22%`) | **0** mẫu (`0.00%`) |
| Trùng vector HOG (Val vs Fit) | **128** mẫu (`2.22%`) | **0** mẫu (`0.00%`) |
| Trung vị cosine similarity 1-NN | `0.9828` (1-NN cùng lớp: `99.98%`) | `0.7950` (1-NN cùng lớp: `61.59%`) |
| **Fit Accuracy (`C=1.0`)** | **100.00%** (`23.040 / 23.040` đúng) | **100.00%** (`23.026 / 23.026` đúng) |
| **Validation Accuracy (`C=1.0`)** | **100.00%** (`5.760 / 5.760` đúng, `0` sai) | **65.2234%** (`3.766 / 5.774` đúng, `2.008` sai) |
| **Validation Macro Precision** | **100.00%** | **69.7273%** |
| **Validation Macro Recall** | **100.00%** | **65.1103%** |
| **Validation Macro F1** | **100.00%** | **63.3226%** |
| Thời gian fit / predict | `130.318s` / `0.064s` | `73.334s` / `0.029s` |

- Model mới được lưu riêng tại: [svm_baseline_c1_prefix_group.joblib](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/models/svm/svm_baseline_c1_prefix_group.joblib)
- Báo cáo JSON mới được lưu riêng tại: [svm_baseline_c1_prefix_group_validation.json](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_baseline_c1_prefix_group_validation.json)
- Tập chỉ số và metadata split mới: [svm_prefix_group_split_indices.npz](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_prefix_group_split_indices.npz), [svm_prefix_group_split_metadata.json](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_prefix_group_split_metadata.json)

---

## 4. Hạn chế của tập Test gốc và lưu ý cho toàn nhóm

- Tập `test` gốc (7.200 mẫu trong `asl_processed/test`) **cũng chứa cả 10 nhóm tiền tố `P1..P10`** (mỗi nhóm từ 674 đến 762 ảnh, phủ toàn bộ 360 cụm `(P_k, lớp)` xen kẽ số thứ tự với tập `train`).
- Do đó:
  1. Việc chuyển `validation` sang chia theo nhóm tiền tố giúp quá trình **chọn siêu tham số `C` công bằng, phân biệt rõ mức độ tổng quát hóa giữa các cấu hình `C`** (thay vì mọi cấu hình đều bão hòa ở mức ~100% trên validation ngẫu nhiên).
  2. Tuy nhiên, việc sửa validation **không làm cho tập `test` gốc trở thành tập đánh giá nhóm tiền tố mới hay người mới**.
  3. Chúng ta **không tự ý thay đổi tập `test` gốc** để đảm bảo cả nhóm đánh giá trên cùng một tập `test` chuẩn của đồ án, nhưng cần nêu rõ đặc điểm này trong báo cáo chung của nhóm khi so sánh kết quả giữa validation theo nhóm và test gốc.
