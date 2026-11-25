
import numpy as np


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Thực hiện phép tích chập 2D trên ảnh xám.

    Ảnh được đệm biên bằng cách lặp lại giá trị pixel biên.
    Kết quả trả về là mảng float32 để giữ kết quả tính toán.
    """
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    if image.ndim != 2:
        raise ValueError("Ảnh đầu vào phải là ảnh xám 2D.")

    if image.size == 0:
        raise ValueError("Ảnh đầu vào không được rỗng.")

    if kernel.ndim != 2 or kernel.size == 0:
        raise ValueError("Kernel phải là ma trận 2D không rỗng.")

    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Ảnh phải chứa dữ liệu số.")

    if not np.issubdtype(kernel.dtype, np.number):
        raise ValueError("Kernel phải chứa dữ liệu số.")

    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(kernel)):
        raise ValueError("Ảnh và kernel không được chứa NaN hoặc vô cực.")

    height, width = image.shape
    kernel_height, kernel_width = kernel.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kích thước kernel phải là số lẻ.")

    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = np.pad(
        image.astype(np.float32),
        ((pad_y, pad_y), (pad_x, pad_x)),
        mode="edge",
    )

    # Tích chập đảo kernel theo cả hai chiều.
    flipped_kernel = np.flip(kernel.astype(np.float32), axis=(0, 1))

    result = np.zeros((height, width), dtype=np.float32)

    # Chỉ lặp trên kernel nhỏ; mỗi lát cắt xử lý toàn ảnh bằng NumPy.
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            result += flipped_kernel[kernel_y, kernel_x] * padded[
                kernel_y:kernel_y + height,
                kernel_x:kernel_x + width,
            ]

    return result


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Tạo kernel Gaussian 2D đã chuẩn hóa tổng bằng 1.
    """
    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise ValueError("size phải là số nguyên dương lẻ.")

    if size <= 0 or size % 2 == 0:
        raise ValueError("size phải là số nguyên dương lẻ.")

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma phải là số hữu hạn lớn hơn 0.")

    center = size // 2
    coordinates = np.arange(size, dtype=np.float32) - center
    xx, yy = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(
        -(xx**2 + yy**2) / (2 * sigma**2)
    )

    kernel /= np.sum(kernel)

    return kernel.astype(np.float32)


def gaussian_filter(
    image: np.ndarray,
    size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Làm mượt ảnh xám bằng Gaussian Filter.

    Kết quả được làm tròn và giới hạn trong khoảng 0–255.
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("Ảnh đầu vào phải là ảnh xám 2D.")

    if image.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu uint8.")

    kernel = gaussian_kernel(size, sigma)
    filtered = convolve2d(image, kernel)

    return np.clip(
        np.rint(filtered), 0, 255
    ).astype(np.uint8)


def bilateral_filter(
    image: np.ndarray,
    size: int = 5,
    sigma_space: float = 2.0,
    sigma_color: float = 50.0,
) -> np.ndarray:
    """
    Làm mượt ảnh xám bằng Bilateral Filter,
    cố gắng bảo toàn các đường biên.

    Args:
        image: Ảnh xám uint8, shape (height, width).
        size: Kích thước cửa sổ lọc, số lẻ dương.
        sigma_space: Độ ảnh hưởng của khoảng cách không gian.
        sigma_color: Độ ảnh hưởng của chênh lệch cường độ pixel.

    Returns:
        Ảnh sau lọc, kiểu uint8.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image phải là NumPy array")

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("image phải là ảnh xám uint8")

    if not isinstance(size, (int, np.integer)) or isinstance(size, bool):
        raise TypeError("size phải là số nguyên")

    if size <= 0 or size % 2 == 0:
        raise ValueError("size phải là số lẻ dương")

    for name, value in (
        ("sigma_space", sigma_space),
        ("sigma_color", sigma_color),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise TypeError(f"{name} phải là số")

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} phải là số hữu hạn dương")

    radius = size // 2

    source = image.astype(np.float32)
    padded = np.pad(source, pad_width=radius, mode="edge")

    # Tạo trọng số không gian: pixel càng xa thì trọng số càng nhỏ.
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)

    distance_squared = xx**2 + yy**2

    spatial_weights = np.exp(
        -distance_squared / (2.0 * sigma_space**2)
    )

    height, width = image.shape
    weighted_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    # Chỉ lặp theo cửa sổ lọc; mọi pixel được xử lý đồng thời bằng NumPy.
    # Cách này nhanh hơn đáng kể so với hai vòng lặp Python theo H và W.
    for offset_y in range(size):
        for offset_x in range(size):
            neighbor = padded[
                offset_y:offset_y + height,
                offset_x:offset_x + width,
            ]
            difference = neighbor - source
            color_weight = np.exp(
                -(difference**2) / (2.0 * sigma_color**2)
            )
            weight = spatial_weights[offset_y, offset_x] * color_weight
            weighted_sum += weight * neighbor
            weight_sum += weight

    filtered = weighted_sum / np.maximum(weight_sum, np.finfo(np.float32).eps)
    return np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
