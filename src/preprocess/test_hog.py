from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from skimage.feature import hog


# Những định dạng ảnh được chương trình chấp nhận
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Mọi ảnh sẽ được chuẩn hóa về 64 x 64
TARGET_SIZE = (64, 64)


def find_first_image(folder: Path):
    """
    Tìm ảnh đầu tiên trong một thư mục
    và toàn bộ các thư mục con của nó.
    """
    for image_path in sorted(folder.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            return image_path

    return None


def resize_with_padding(image, target_size=(64, 64)):
    """
    Thu nhỏ ảnh nhưng vẫn giữ nguyên tỷ lệ gốc.

    Phần diện tích còn thiếu được thêm nền trắng
    để ảnh cuối cùng có kích thước 64 x 64.
    """
    target_width, target_height = target_size
    original_height, original_width = image.shape[:2]

    # Tính tỷ lệ thu nhỏ sao cho ảnh vừa khung 64 x 64
    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    new_width = max(1, round(original_width * scale))
    new_height = max(1, round(original_height * scale))

    # Thu nhỏ ảnh nhưng không làm sai tỷ lệ
    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    # Tạo một ảnh nền trắng kích thước 64 x 64
    canvas = np.full(
        (target_height, target_width, 3),
        255,
        dtype=np.uint8,
    )

    # Tính vị trí để đặt ảnh bàn tay vào chính giữa
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Dán ảnh đã thu nhỏ lên nền trắng
    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width,
    ] = resized_image

    return canvas


# Kiểm tra người dùng đã truyền đường dẫn dataset hay chưa
if len(sys.argv) < 2:
    print("Vui long truyen duong dan thu muc asl_processed.")
    print(
        'Vi du: python src/preprocess/test_hog.py '
        '"C:\\duong-dan\\asl_processed"'
    )
    raise SystemExit(1)


# Nhận đường dẫn dataset từ Terminal
data_root = Path(sys.argv[1])
train_dir = data_root / "train"


# Kiểm tra thư mục train có tồn tại hay không
if not train_dir.exists():
    print(f"Khong tim thay thu muc train: {train_dir}")
    raise SystemExit(1)


# Tìm một ảnh đầu tiên trong tập train
image_path = find_first_image(train_dir)

if image_path is None:
    print("Khong tim thay anh trong tap train.")
    raise SystemExit(1)


# OpenCV đọc ảnh màu
original_image = cv2.imread(str(image_path))

if original_image is None:
    print(f"Khong doc duoc anh: {image_path}")
    raise SystemExit(1)


# Resize về 64 x 64 nhưng không làm méo bàn tay
resized_image = resize_with_padding(
    original_image,
    TARGET_SIZE,
)


# Chuyển ảnh màu thành ảnh xám
gray_image = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2GRAY,
)


# Trích xuất đặc trưng HOG
hog_features, hog_image = hog(
    gray_image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    feature_vector=True,
)


# Làm ảnh HOG sáng và dễ quan sát hơn
if hog_image.max() > 0:
    hog_image = exposure.rescale_intensity(
        hog_image,
        in_range=(0, hog_image.max()),
    )


# In thông tin ra Terminal
print(f"Duong dan anh: {image_path}")
print(f"Nhan cua anh: {image_path.parent.name}")

print(
    "Kich thuoc anh goc: "
    f"{original_image.shape[1]} x {original_image.shape[0]}"
)

print(
    "Kich thuoc sau resize va padding: "
    f"{resized_image.shape[1]} x {resized_image.shape[0]}"
)

print(f"So dac trung HOG: {len(hog_features)}")

print("10 dac trung HOG dau tien:")
print(hog_features[:10])


# OpenCV dùng thứ tự màu BGR,
# trong khi Matplotlib sử dụng RGB
rgb_original_image = cv2.cvtColor(
    original_image,
    cv2.COLOR_BGR2RGB,
)

rgb_resized_image = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2RGB,
)


# Hiển thị quá trình tiền xử lý
fig, axes = plt.subplots(1, 4, figsize=(15, 4))

axes[0].imshow(rgb_original_image)
axes[0].set_title("Anh goc")
axes[0].axis("off")

axes[1].imshow(rgb_resized_image)
axes[1].set_title("Resize + padding")
axes[1].axis("off")

axes[2].imshow(gray_image, cmap="gray")
axes[2].set_title("Anh xam 64 x 64")
axes[2].axis("off")

axes[3].imshow(hog_image, cmap="gray")
axes[3].set_title("Dac trung HOG")
axes[3].axis("off")

plt.tight_layout()
plt.show()