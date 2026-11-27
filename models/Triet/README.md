# Nhận diện Ngôn ngữ ký hiệu Mỹ — ASL Alphabet A-Z

Đồ án nhận diện 26 ký tự bàn tay ASL bằng MediaPipe Hands, đặc trưng hình học 91 chiều và các mô hình phân loại TensorFlow/Keras và scikit-learn. Pipeline hỗ trợ ảnh tĩnh, tập dữ liệu Kaggle/Mendeley, webcam thực tế và ứng dụng desktop.

## Kiến trúc tổng quan

```text
Ảnh hoặc frame webcam
        |
        v
MediaPipe Hands -> 21 landmark (x, y, z)
        |
        v
Dời gốc về cổ tay -> chia d_max -> vector 91 chiều
        |
        v
MLP / SVM RBF / KNN / Random Forest
        |
        v
Xác suất 26 lớp -> EMA -> margin check -> sliding window
```

Vector 91 chiều gồm:

- 63 tọa độ landmark;
- 15 góc gập đốt ngón;
- 10 khoảng cách đầu ngón;
- 3 thành phần pháp tuyến lòng bàn tay.

## Cấu trúc thư mục

| Thư mục | Vai trò |
|---|---|
| `configs/` | Cấu hình đường dẫn, MediaPipe, chia tập, MLP và hậu xử lý |
| `datasets/` | Dữ liệu raw, metadata và Parquet đã trích đặc trưng |
| `src/data/` | Quét dữ liệu, deduplicate SHA-256 và chia stratified train/val/test |
| `src/preprocessing/` | MediaPipe Hands và chuẩn hóa landmark |
| `src/feature_extraction/` | Rút góc, khoảng cách, pháp tuyến và tọa độ |
| `src/models/` | MLP Keras và các mô hình scikit-learn |
| `src/training/` | Augmentation, callbacks và pipeline huấn luyện |
| `src/evaluation/` | Metric, confusion matrix, learning curve và FPS |
| `src/inference/` | Lọc EMA/margin/sliding window cho webcam |
| `scripts/` | Entry point build dataset, train và evaluate |
| `app/` | Giao diện CustomTkinter |
| `models/` | Checkpoint và model production |
| `tests/` | Kiểm thử pytest |
| `reports/` | Log, metric, hình và confusion matrix |
| `docs/` | Tài liệu tiếp cận mô hình |

## Cài đặt

Yêu cầu Python 3.10–3.12; khuyến nghị Python 3.11.

### Venv

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Conda

```powershell
conda create -n asl python=3.11
conda activate asl
python -m pip install -r requirements.txt
```

Kiểm tra cấu hình:

```powershell
python scripts/check_config.py
python -m pytest tests
```

## Chuẩn bị dữ liệu

Đặt ảnh `.jpg`, `.jpeg` hoặc `.png` theo thư mục nhãn A-Z:

```text
datasets/raw/kaggle_asl/A/...
datasets/raw/kaggle_asl/B/...
datasets/raw/mendeley_asl/A/...
datasets/raw/webcam_real_test/A/...
```

Không bỏ ảnh trực tiếp ở thư mục root của `kaggle_asl` hoặc `mendeley_asl`.

## Quy trình chạy

Chạy theo thứ tự:

```powershell
python scripts/run_create_manifest.py
python scripts/run_build_dataset.py
python -m pytest tests
python scripts/run_train.py --exp exp1
python scripts/run_train.py --exp exp2
python scripts/run_train.py --exp exp3
python scripts/run_evaluate.py
python app/main_window.py
```

Exp1 sử dụng KNN và MLP 63 chiều. Exp2 dùng MLP 91 chiều. Exp3 dùng MLP tối ưu và SVM RBF có hiệu chỉnh xác suất.

Lưu model và kết quả tại:

- `models/production/*.keras`
- `models/production/*.onnx`
- `models/production/*.joblib`
- `models/production/*_metadata.json`
- `reports/metrics/benchmark_summary.csv`
- `reports/figures/`

## Xử lý lỗi thường gặp

### Không tìm thấy ảnh

Chạy preflight trước:

```powershell
python scripts/run_create_manifest.py
```

Nếu báo `PREFLIGHT FAILED`, kiểm tra dữ liệu có nằm trong thư mục con `A-Z` và đúng phần mở rộng ảnh hay không.

### Camera mở sai thiết bị

Thay `camera_id` trong `app/main_window.py` hoặc truyền camera ID tùy mã khởi tạo ứng dụng. Thông thường thử `0`, `1`, `2`.

### NumPy/MediaPipe import lỗi DLL

Đảm bảo dùng Python 3.11 và cài lại environment:

```powershell
python -m pip uninstall -y numpy mediapipe tensorflow protobuf
python -m pip install -r requirements.txt
```

### ONNX Runtime hoặc TensorFlow crash

Không đồng thời đổi phiên bản các package bằng pip. Tạo lại `.venv`, cài đúng `requirements.txt` rồi mở terminal mới.

### Thiếu Parquet

Chạy lại:

```powershell
python scripts/run_create_manifest.py
python scripts/run_build_dataset.py
```

## Tài liệu mô hình

Xem [docs/MODEL_GUIDE.md](docs/MODEL_GUIDE.md) để theo dõi dữ liệu đi qua MediaPipe, vector đặc trưng, mạng MLP và suy luận model.
