# Báo cáo Thực nghiệm Siêu tham số `C` cho LinearSVC (Đánh giá theo Nhóm Tiền Tố)

## 1. Thiết lập thực nghiệm

- **Mô hình**: Cùng một loại mô hình **`LinearSVC`** (`penalty="l2"`, `loss="squared_hinge"`, `dual="auto"`, `random_state=42`, `max_iter=10000`) đặt trong `Pipeline`, không dùng scaler. Đây là **4 cấu hình siêu tham số `C` của cùng một họ mô hình SVM tuyến tính**, không phải 4 thuật toán khác nhau.
- **Đặc trưng**: 1764 đặc trưng HOG (`64×64`, `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `block_norm="L2-Hys"`) từ `data/processed/svm_hog_features.npz` (SHA-256: `096667c46efa4a3d160043fc1c5c32ce9febf1c910d179a2a0191f036db16402`).
- **Tập chia (Split)**: Sử dụng nguyên vẹn tập chỉ số [svm_prefix_group_split_indices.npz](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/results/svm/svm_prefix_group_split_indices.npz):
  - **Fit**: `23.026` mẫu thuộc 8 nhóm tiền tố tên file `P1, P2, P3, P4, P5, P6, P7, P9` (đủ 36 lớp).
  - **Validation**: `5.774` mẫu thuộc 2 nhóm tiền tố tên file `P8, P10` (đủ 36 lớp).
  - **Kiểm chứng độc lập**: `0` chỉ số dòng giao nhau, `0` nhóm tiền tố giao nhau, `0` ảnh trùng SHA-256 và `0` vector HOG trùng giữa `fit` và `validation`.
  - **Tập Test (`X_test`, `y_test`)**: Hoàn toàn không được nạp hay sử dụng trong quá trình chọn `C`.
- **Tái sử dụng baseline `C = 1.0`**: Sau khi kiểm chứng khớp 100% fingerprint dữ liệu, chỉ số split và cấu hình `LinearSVC`, kết quả của `C = 1.0` được tái sử dụng từ [svm_baseline_c1_prefix_group.joblib](file:///C:/Users/Quang%20Diep/Machine-Learning---Sign-Language-Recognition/models/svm/svm_baseline_c1_prefix_group.joblib) (và tính bổ sung thông tin số vòng lặp `n_iter_max = 27`, đồng thời lưu bản sao thực nghiệm riêng tại `models/svm/svm_c1_0_prefix_group.joblib`).

---

## 2. Bảng kết quả thực tế của 4 giá trị `C`

| Giá trị `C` | Fit Accuracy (Đúng/Tổng) | Val Accuracy (Chưa làm tròn) | Val Đúng / Sai (trên 5.774) | Val Macro Precision | Val Macro Recall | **Val Macro F1 (Chưa làm tròn)** | Thời gian Fit / Predict | Hội tụ (`n_iter_max`, Cảnh báo) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0.01`** | `100.00%` (`23026/23026`) | `0.654659` (`65.47%`) | `3.780` / `1.994` | `0.722645` (`72.26%`) | `0.653407` (`65.34%`) | **`0.636364`** (**`63.64%`**) | `96.714s` / `0.037s` | Hội tụ (`14` vòng, `0` cảnh báo) |
| `0.1` | `100.00%` (`23026/23026`) | `0.655178` (`65.52%`) | `3.783` / `1.991` | `0.718496` (`71.85%`) | `0.654078` (`65.41%`) | `0.635587` (`63.56%`) | `91.855s` / `0.056s` | Hội tụ (`17` vòng, `0` cảnh báo) |
| `1.0` *(Baseline)* | `100.00%` (`23026/23026`) | `0.652234` (`65.22%`) | `3.766` / `2.008` | `0.697273` (`69.73%`) | `0.651103` (`65.11%`) | `0.633226` (`63.32%`) | `73.334s` / `0.029s` | Hội tụ (`27` vòng, `0` cảnh báo) |
| `10.0` | `100.00%` (`23026/23026`) | `0.653793` (`65.38%`) | `3.775` / `1.999` | `0.701511` (`70.15%`) | `0.652454` (`65.25%`) | `0.635365` (`63.54%`) | `99.454s` / `0.049s` | Hội tụ (`21` vòng, `0` cảnh báo) |

---

## 3. Lựa chọn `C` tối ưu và phân tích kết quả

- **Quy tắc chọn**: Chọn `C` có **Validation Macro F1 (chưa làm tròn) cao nhất**; nếu bằng nhau thì ưu tiên `C` nhỏ hơn; không chọn theo `Fit Accuracy`.
- **Kết quả chọn**: **`C = 0.01`** là giá trị tốt nhất trong 4 mức đã thử nghiệm trên split nhóm tiền tố này:
  - **Validation Macro F1**: đạt **`0.636364`** (`63.64%`), cao nhất trong 4 mức `C` và **tăng `+0.3138` điểm phần trăm** (`+0.003138`) so với baseline `C = 1.0` (`0.633226`).
  - **Validation Macro Precision**: đạt **`0.722645`** (`72.26%`), **tăng `+2.5372` điểm phần trăm** so với baseline `C = 1.0` (`0.697273`).
  - **Validation Accuracy**: đạt **`0.654659`** (`65.47%`), **tăng `+0.2425` điểm phần trăm** so với baseline `C = 1.0` (`0.652234`), tức dự đoán đúng thêm **14 mẫu** trên tập validation (`3.780` so với `3.766` mẫu).
  - *(Lưu ý trung thực)*: Mức `C = 0.1` có Validation Accuracy nhỉnh hơn `C = 0.01` đúng 3 mẫu (`3.783` so với `3.780`, tức `65.52%` so với `65.47%`), nhưng theo đúng tiêu chí đã chốt trước thực nghiệm là **Validation Macro F1 chưa làm tròn**, `C = 0.01` đạt `0.636364 > 0.635587` nên `C = 0.01` được chọn một cách khách quan. Đồng thời, mức cải thiện giữa 4 giá trị `C` là khá nhỏ (`~0.31` điểm phần trăm Macro F1), vì với không gian 1764 chiều HOG, cả 4 mức `C` đều đã phân tách tuyến tính 100% tập `fit` và hội tụ rất nhanh chỉ sau 14–27 vòng lặp.
