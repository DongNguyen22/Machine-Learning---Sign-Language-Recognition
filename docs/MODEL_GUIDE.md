# Hướng dẫn tiếp cận mô hình

Tài liệu này mô tả pipeline từ ảnh đầu vào đến xác suất lớp ASL và cách kiểm thử model đã lưu.

## Bước 1 — Dữ liệu đi qua mô hình

```text
Ảnh RGB            (H, W, 3)
       |
       v
MediaPipe Hands    (21, 3)
       |
       v
Chuẩn hóa          (21, 3), wrist = 0
                   chia theo d_max
       |
       v
Vector 91 chiều    (91,)
  63 tọa độ
  15 góc
  10 khoảng cách
  3 pháp tuyến
       |
       v
Softmax hoặc predict
Xác suất 26 lớp    (26,)
```

Cổ tay được đặt về `(0, 0, 0)` để kết quả ít phụ thuộc vị trí tay trong khung hình. `d_max` là khoảng cách Euclid lớn nhất từ cổ tay tới một landmark. Việc chia toàn bộ vector cho `d_max` giúp kết quả ổn định khi tay gần hoặc xa camera.

Trong 15 góc, mỗi ngón có ba góc tại khớp liên tiếp. Góc được tính bằng `arccos` của tích vô hướng hai vector đốt kề nhau. 10 khoảng cách gồm năm khoảng từ đầu ngón đến cổ tay và năm khoảng giữa các đầu ngón. Vector pháp tuyến lấy từ tích có hướng của `(P5-P0)` và `(P17-P0)`, rồi chuẩn hóa đơn vị.

## Bước 2 — Cấu trúc mô hình

### MLP baseline

```text
63 input -> Dense(64, ReLU) -> Dense(26, Softmax)
```

Baseline dùng 63 tọa độ làm điều kiện đối chứng đơn giản.

### MLP geometric

```text
91 input -> Dense(256) -> BatchNorm -> ReLU -> Dropout(0.3)
         -> Dense(128) -> BatchNorm -> ReLU -> Dropout(0.2)
         -> Dense(64, ReLU) -> Dense(26, Softmax)
```

`BatchNormalization` ổn định phân phối hoạt động giữa các lớp. `Dropout` giảm phụ thuộc vào landmark cụ thể. `Label Smoothing 0.05` làm mô hình không quá tin tuyệt đối vào một nhãn.

### Mô hình đối chứng

- KNN `k=5`: dự đoán theo láng giềng gần nhất trong không gian vector.
- SVM RBF: tạo ranh giới phi tuyến; pipeline có `StandardScaler` và calibrator.
- Random Forest: 200 cây; hữu ích để kiểm tra mô hình ensemble không phụ thuộc scaling.

## Bước 3 — Inference tối giản

Ví dụ nạp model `.keras` và dự đoán 91 đặc trưng đã chuẩn bị sẵn:

```python
import cv2
import numpy as np
from tensorflow import keras

from src.preprocessing.hand_detector import HandDetector
from src.preprocessing.coordinate_normalizer import Normalizer
from src.feature_extraction.geometric_features import FeatureExtractor

labels = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
model = keras.models.load_model("models/production/exp3_mlp_opt.keras")
detector = HandDetector(static_image_mode=True)
normalizer = Normalizer()
extractor = FeatureExtractor()

image = cv2.imread("path/to/hand.jpg")
landmarks = detector.process(image)
if landmarks is not None:
    normalized = normalizer.transform(landmarks)
    features = extractor.transform(normalized).reshape(1, -1)
    probabilities = model.predict(features, verbose=0)[0]
    index = int(np.argmax(probabilities))
    print(labels[index], float(probabilities[index]))

detector.close()
```

Đây là luồng inference tối thiểu. Ứng dụng webcam bổ sung EMA, margin và sliding window để giảm rung nhãn.

## Bước 4 — Chạy nhận diện bằng webcam

Tại thư mục gốc dự án:

```powershell
cd D:\asl-alphabet-recognition
.\.venv\Scripts\Activate.ps1
python app\main_window.py
```

Trước khi mở ứng dụng, cần có ít nhất một model trong `models/production/`. Hệ thống ưu tiên `exp3_mlp_opt.keras`, sau đó kiểm tra các model khác theo thứ tự trong `app/main_window.py`. Chạy `python scripts/run_train.py --exp exp3` trước nếu thư mục production chưa có model.

Camera mặc định là `camera_id=0`. Nếu ứng dụng báo `Không mở được camera`, kiểm tra camera bằng Windows Settings hoặc thử đổi trong `app/main_window.py`:

```python
window = ASLWindow(camera_id=1)
```

Các thiết bị có thể là `0`, `1` hoặc `2` tùy thứ tự hệ điều hành nhận camera.

Trong khi chạy:

- luồng nền đọc camera và MediaPipe để giao diện không bị chặn;
- mỗi frame được chuẩn hóa và chuyển thành vector 91 chiều;
- MLP/joblib trả xác suất 26 ký tự;
- `EMA` làm mượt xác suất;
- `Margin Check` loại kết quả hai nhãn cao nhất quá gần nhau;
- `Sliding Window` chỉ hiển thị ký tự khi đạt số phiếu tối thiểu.

Các ngưỡng hiện tại nằm trong `configs/config.yaml`, mục `inference.postprocess`.

## Bước 5 — Điều chỉnh và mở rộng

Các ngưỡng nằm trong `configs/config.yaml`:

- `conf_threshold`: tăng để giảm dự đoán bồi; giảm nếu ký hiệu hợp lệ bị bỏ.
- `margin_threshold`: tăng để chỉ nhận nhãn có độ phân biệt rõ.
- `window_size` và `min_votes`: tăng để chống rung, giảm để phản hồi nhanh hơn.
- `ema_alpha`: alpha lớn hơn bám xác suất mới nhanh hơn nhưng dễ rung.

Khi đổi môi trường ánh sáng, bắt đầu bằng:

1. Giữ margin hiện tại, tăng hoặc giảm conf threshold.
2. Thu thập test riêng cho điều kiện ánh sáng mới.
3. Chỉ thay threshold sau khi kiểm tra cả tập in-domain và webcam.

Khi mở rộng ký tự:

1. Cập nhật `configs/label_map.json`.
2. Cập nhật `num_classes` trong `config.yaml`.
3. Chia dữ liệu mới bằng `run_create_manifest.py`.
4. Huấn luyện lại toàn bộ thí nghiệm.
5. Cập nhật ma trận 26x26 thành kích thước lớp mới.
6. Chạy pytest và evaluator trước khi thay model production.
