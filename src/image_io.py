
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def normalize_image(image: Image.Image) -> Image.Image:
    """Chuẩn hóa ảnh Pillow bất kỳ thành RGB, có xét EXIF và transparency."""
    image = ImageOps.exif_transpose(image)

    # Ảnh động được xử lý ở frame đầu tiên, phù hợp với đầu ra ảnh tĩnh.
    if getattr(image, "is_animated", False):
        image.seek(0)

    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    return image.convert("RGB")


def resize_for_processing(
    image: Image.Image,
    max_pixels: int = 1_500_000,
) -> Image.Image:
    """Giảm ảnh quá lớn, giữ nguyên tỉ lệ; ảnh nhỏ không bị thay đổi."""
    if (
        not isinstance(max_pixels, int)
        or isinstance(max_pixels, bool)
        or max_pixels <= 0
    ):
        raise ValueError("max_pixels phải là số nguyên dương.")

    width, height = image.size
    pixel_count = width * height
    if pixel_count <= max_pixels:
        return image.copy()

    scale = np.sqrt(max_pixels / pixel_count)
    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def read_image(image_path: str | Path) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn và trả về mảng RGB.

    Args:
        image_path: Đường dẫn đến ảnh đầu vào.

    Returns:
        Mảng NumPy có dạng (height, width, 3), kiểu uint8.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")

    with Image.open(path) as image:
        return np.array(normalize_image(image), dtype=np.uint8)


def save_image(image_array: np.ndarray, output_path: str | Path) -> None:
    """
    Lưu mảng ảnh NumPy thành tệp ảnh.
    """
    image_array = np.asarray(image_array)

    if image_array.dtype != np.uint8:
        raise ValueError("Ảnh cần được chuyển về kiểu uint8 trước khi lưu.")

    if image_array.ndim not in (2, 3):
        raise ValueError("Ảnh phải là ảnh xám hoặc ảnh màu.")

    if image_array.ndim == 3 and image_array.shape[2] not in (3, 4):
        raise ValueError("Ảnh màu phải có 3 kênh RGB hoặc 4 kênh RGBA.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image_array).save(path)
