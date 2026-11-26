from pathlib import Path
import sys

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_first_image(folder: Path):
    """Tìm ảnh đầu tiên trong thư mục và các thư mục con."""
    for image_path in sorted(folder.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            return image_path

    return None


def inspect_image(image_path: Path, split_name: str):
    """Đọc, in thông tin và hiển thị một ảnh."""
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Khong doc duoc anh: {image_path}")
        return

    height, width, channels = image.shape

    print(f"\n--- {split_name.upper()} ---")
    print(f"Duong dan: {image_path}")
    print(f"Nhan: {image_path.parent.name}")
    print(f"Kich thuoc: {width} x {height}")
    print(f"So kenh mau: {channels}")

    cv2.imshow(f"Anh mau {split_name}", image)


if len(sys.argv) < 2:
    print("Vui long truyen duong dan thu muc asl_processed.")
    raise SystemExit(1)


data_root = Path(sys.argv[1])
train_dir = data_root / "train"
test_dir = data_root / "test"


if not train_dir.exists():
    print(f"Khong tim thay thu muc train: {train_dir}")
    raise SystemExit(1)

if not test_dir.exists():
    print(f"Khong tim thay thu muc test: {test_dir}")
    raise SystemExit(1)


train_image_path = find_first_image(train_dir)
test_image_path = find_first_image(test_dir)

if train_image_path is None:
    print("Khong tim thay anh trong tap train.")
    raise SystemExit(1)

if test_image_path is None:
    print("Khong tim thay anh trong tap test.")
    raise SystemExit(1)


inspect_image(train_image_path, "train")
inspect_image(test_image_path, "test")

print("\nNhan phim bat ky tren cua so anh de dong.")

cv2.waitKey(0)
cv2.destroyAllWindows()