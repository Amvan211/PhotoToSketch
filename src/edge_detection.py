
import numpy as np
from src.filters import convolve2d


def sobel_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính gradient theo hai hướng bằng toán tử Sobel.

    Args:
        image: Ảnh xám uint8, shape (height, width).

    Returns:
        Gx, Gy: Hai mảng gradient kiểu float32.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image phải là NumPy array")

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image phải là ảnh xám uint8")

    kernel_x = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    )

    kernel_y = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    )

    gx = convolve2d(image, kernel_x)
    gy = convolve2d(image, kernel_y)

    return gx, gy


def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Phát hiện biên bằng Sobel.

    Returns:
        Ảnh biên uint8, giá trị nằm trong khoảng 0-255.
    """
    gx, gy = sobel_gradients(image)

    magnitude = np.sqrt(gx**2 + gy**2)

    # Dùng percentile thay cho giá trị cực đại để một vài cạnh rất mạnh
    # không làm các nét còn lại trở nên quá nhạt.
    positive_magnitudes = magnitude[magnitude > 0]
    if positive_magnitudes.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    scale_value = float(np.percentile(positive_magnitudes, 99.0))

    normalized = magnitude * (255.0 / max(scale_value, 1e-6))

    return np.clip(
        np.rint(normalized), 0, 255
    ).astype(np.uint8)
