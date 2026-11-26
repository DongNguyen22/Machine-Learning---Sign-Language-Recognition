from pathlib import Path
import argparse

import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_with_padding(image, target_size=(64, 64)):
    """
    Thu nhỏ ảnh nhưng vẫn giữ nguyên tỉ lệ.
    Phần còn trống được tô màu trắng.
    """
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (target_height, target_width, 3),
        255,
        dtype=np.uint8
    )

    x_start = (target_width - new_width) // 2
    y_start = (target_height - new_height) // 2

    canvas[
        y_start:y_start + new_height,
        x_start:x_start + new_width
    ] = resized_image

    return canvas


def read_image_bgr(image_path):
    """
    Đọc ảnh màu BGR (uint8) từ đường dẫn bất kỳ trên Windows,
    hỗ trợ cả đường dẫn có dấu cách và ký tự tiếng Việt (Unicode).
    Dùng cùng bộ giải mã cv2.IMREAD_COLOR như cv2.imread.
    """
    path_obj = Path(image_path)
    if not path_obj.is_file():
        return None
    try:
        raw_bytes = path_obj.read_bytes()
    except OSError:
        return None
    if not raw_bytes:
        return None
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(byte_array, cv2.IMREAD_COLOR)


def suppress_background_preserve_hand(image_bgr):
    """
    Giảm ảnh hưởng của nền phòng tối/nhiễu sọc nhưng giữ nguyên vẹn 100% ngón tay
    và đường viền bàn tay (Cách 3B đã kiểm chứng trong scripts/compare_three_input_pipelines.py):
    1. Xác định vùng lõi màu da trong không gian YCrCb (Cr >= 131, 77 <= Cb <= 138, Y >= 45).
    2. Đóng hình thái học (MORPH_CLOSE 11x11, 2 lần) để lấp kín các vùng chói sáng trên ngón tay.
    3. Giữ các thành phần liên thông giao với vùng trung tâm 70% của khung crop (diện tích >= 60 px).
    4. Nới rộng biên bảo vệ (dilate 9x9) để không bao giờ cắt lẹm vào đầu ngón tay hay viền bàn tay.
    5. Hòa trộn biên mềm (GaussianBlur 13x13, sigma=3.5), nâng vùng nền ngoài bàn tay về mức xám sáng 232
       tương đồng với phông nền sáng của tập huấn luyện.
    """
    h, w = image_bgr.shape[:2]
    ycc = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0]
    Cr = ycc[:, :, 1]
    Cb = ycc[:, :, 2]

    skin_raw = ((Cr >= 131) & (Cb >= 77) & (Cb <= 138) & (Y >= 45)).astype(np.uint8) * 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_closed = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, k_close, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skin_closed, connectivity=8)
    inner_y1, inner_y2 = int(h * 0.15), int(h * 0.85)
    inner_x1, inner_x2 = int(w * 0.15), int(w * 0.85)
    clean_mask = np.zeros_like(skin_closed)
    for lbl_id in range(1, num_labels):
        comp = labels == lbl_id
        if np.any(comp[inner_y1:inner_y2, inner_x1:inner_x2]) and stats[lbl_id, cv2.CC_STAT_AREA] >= 60:
            clean_mask[comp] = 255

    if np.count_nonzero(clean_mask) == 0:
        clean_mask = skin_closed

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    protected_hand_mask = cv2.dilate(clean_mask, k_dilate, iterations=1)

    alpha = cv2.GaussianBlur(protected_hand_mask.astype(np.float32) / 255.0, (13, 13), 3.5)
    bg_color = np.full_like(image_bgr, 232, dtype=np.float32)
    blended = np.clip(
        image_bgr.astype(np.float32) * alpha[:, :, None] + bg_color * (1.0 - alpha[:, :, None]),
        0,
        255,
    ).astype(np.uint8)
    return blended, protected_hand_mask


def apply_webcam_method_3b_from_bgr(image_bgr, already_flipped=False):
    """
    Áp dụng Cách 3B cho ảnh crop từ webcam:
    - Lật ngang ảnh đúng 1 lần so với khung hình camera gốc (nếu already_flipped=True thì
      giữ nguyên, tuyệt đối không lật lần thứ hai).
    - Giảm ảnh hưởng của nền nhưng giữ nguyên viền bàn tay bằng suppress_background_preserve_hand().
    Trả về (processed_bgr, flipped_once_bgr, protected_hand_mask).
    """
    flipped_once_bgr = image_bgr if already_flipped else cv2.flip(image_bgr, 1)
    processed_bgr, protected_hand_mask = suppress_background_preserve_hand(flipped_once_bgr)
    return processed_bgr, flipped_once_bgr, protected_hand_mask


def extract_hog_with_diagnostics_from_bgr(image):
    """
    Chạy đúng một pipeline tiền xử lý duy nhất trên ảnh BGR (uint8) trong bộ nhớ:
    1. resize_with_padding về 64x64 (nền trắng 255)
    2. cv2.cvtColor sang ảnh xám (grayscale 64x64 uint8)
    3. skimage.feature.hog trích xuất 1764 đặc trưng (float32)

    Trả về bộ (features_float32, resized_bgr_64x64, gray_uint8_64x64) để chế độ
    chẩn đoán lấy trực tiếp ảnh xám 64x64 thực tế đã đưa vào HOG mà không dựng
    một pipeline riêng.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        return None

    resized_image = resize_with_padding(image, (64, 64))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features.astype(np.float32), resized_image, gray_image


def extract_hog_from_bgr(image):
    """
    Nhận một ảnh màu BGR (uint8) đã nằm trong bộ nhớ (RAM)
    và chuyển thành 1764 đặc trưng HOG (float32).
    Dùng chung cho cả đọc từ file ảnh và khung hình từ camera/API.
    """
    out = extract_hog_with_diagnostics_from_bgr(image)
    if out is None:
        return None
    return out[0]


def extract_hog(image_path):
    """
    Đọc một ảnh từ đường dẫn file và chuyển ảnh thành 1764 đặc trưng HOG.
    """
    image = read_image_bgr(image_path)

    if image is None:
        return None

    return extract_hog_from_bgr(image)


def get_class_names(split_directory):
    """
    Lấy tên các lớp từ tên thư mục, ví dụ: 0, 1, ..., A, B, ...
    """
    return sorted(
        folder.name
        for folder in split_directory.iterdir()
        if folder.is_dir()
    )


def process_split(
    split_directory,
    class_names,
    max_per_class=None
):
    """
    Chuyển toàn bộ ảnh của một tập train hoặc test thành HOG.

    X chứa đặc trưng HOG.
    y chứa số thứ tự của nhãn.
    """
    all_features = []
    all_labels = []
    skipped_images = 0

    for class_index, class_name in enumerate(class_names):
        class_directory = split_directory / class_name

        if not class_directory.exists():
            print(f"Canh bao: khong tim thay lop {class_name}")
            continue

        image_paths = sorted(
            path
            for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if max_per_class is not None:
            image_paths = image_paths[:max_per_class]

        description = f"{split_directory.name}/{class_name}"

        for image_path in tqdm(image_paths, desc=description):
            features = extract_hog(image_path)

            if features is None:
                skipped_images += 1
                print(f"\nKhong doc duoc anh: {image_path}")
                continue

            all_features.append(features)
            all_labels.append(class_index)

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int32)

    return X, y, skipped_images


def main():
    parser = argparse.ArgumentParser(
        description="Chuyen anh ASL thanh dac trung HOG"
    )

    parser.add_argument(
        "dataset_path",
        help="Duong dan den thu muc asl_processed"
    )

    parser.add_argument(
        "--output",
        default="data/processed",
        help="Thu muc luu dac trung HOG"
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Chi xu ly toi da N anh cua moi lop"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    train_directory = dataset_path / "train"
    test_directory = dataset_path / "test"
    output_directory = Path(args.output)

    if not train_directory.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc train: {train_directory}"
        )

    if not test_directory.exists():
        raise FileNotFoundError(
            f"Khong tim thay thu muc test: {test_directory}"
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names(train_directory)

    print(f"So lop tim thay: {len(class_names)}")
    print(f"Danh sach lop: {class_names}")

    print("\nDang xu ly tap TRAIN...")

    X_train, y_train, skipped_train = process_split(
        train_directory,
        class_names,
        args.max_per_class
    )

    print("\nDang xu ly tap TEST...")

    X_test, y_test, skipped_test = process_split(
        test_directory,
        class_names,
        args.max_per_class
    )

    output_path = output_directory / "svm_hog_features.npz"

    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        class_names=np.asarray(class_names)
    )

    print("\n========== KET QUA ==========")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")
    print(f"So anh train bi bo qua: {skipped_train}")
    print(f"So anh test bi bo qua : {skipped_test}")
    print(f"Da luu tai: {output_path.resolve()}")


if __name__ == "__main__":
    main()