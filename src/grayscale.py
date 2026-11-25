
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh RGB sang ảnh xám bằng công thức trọng số.

    Args:
        image: Mảng ảnh RGB có shape (height, width, 3).

    Returns:
        Ảnh xám có shape (height, width), kiểu uint8.
    """
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Đầu vào phải là ảnh RGB có 3 kênh.")

    if image.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu dữ liệu uint8.")

    rgb = image.astype(np.float32)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    gray = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)
