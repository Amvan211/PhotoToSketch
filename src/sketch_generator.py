"""Pipeline chuyển ảnh thành tranh vẽ bằng các kỹ thuật xử lý ảnh cơ bản."""

import numpy as np

from src.edge_detection import sobel_edge_detection
from src.filters import bilateral_filter, gaussian_filter
from src.grayscale import to_grayscale


def _separable_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Làm mờ Gaussian tách 1D để phục vụ Color Dodge."""
    radius = max(1, int(3.0 * sigma))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    source = image.astype(np.float32)

    padded = np.pad(source, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.zeros_like(source)
    for index, weight in enumerate(kernel):
        horizontal += weight * padded[:, index:index + image.shape[1]]

    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    vertical = np.zeros_like(source)
    for index, weight in enumerate(kernel):
        vertical += weight * padded[index:index + image.shape[0], :]
    return np.clip(np.rint(vertical), 0, 255).astype(np.uint8)


def percentile_stretch(
    image: np.ndarray,
    minimum_dynamic_range: float = 120.0,
) -> np.ndarray:
    """Chỉ kéo giãn percentile 1–99 khi ảnh thực sự thiếu tương phản."""
    low, high = np.percentile(image, [1.0, 99.0])
    if high <= low or high - low >= minimum_dynamic_range:
        return image.copy()
    result = (image.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def adjust_tone(
    image: np.ndarray,
    contrast: float = 1.0,
    brightness: float = 0.0,
) -> np.ndarray:
    """Điều chỉnh tuyến tính độ tương phản và độ sáng của ảnh xám."""
    if not np.isfinite(contrast) or contrast <= 0:
        raise ValueError("contrast phải là số hữu hạn lớn hơn 0")
    if not np.isfinite(brightness) or not -100 <= brightness <= 100:
        raise ValueError("brightness phải nằm trong khoảng -100–100")
    result = (image.astype(np.float32) - 127.5) * contrast + 127.5
    result += brightness
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _dodge_blend(image: np.ndarray, sigma: float) -> np.ndarray:
    """Color Dodge an toàn: gray*255/(255-blur(255-gray))."""
    if np.ptp(image) == 0:
        return np.full_like(image, 255)
    inverted = 255 - image
    blurred = _separable_gaussian_blur(inverted, sigma)
    denominator = np.maximum(255.0 - blurred.astype(np.float32), 1.0)
    result = image.astype(np.float32) * 255.0 / denominator
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _edge_layer(
    edges: np.ndarray,
    threshold: float,
    edge_weight: float,
) -> np.ndarray:
    """Đổi độ lớn Sobel thành lớp nét xám mềm trên nền trắng."""
    strength = np.clip(
        (edges.astype(np.float32) - threshold)
        / max(255.0 - threshold, 1.0),
        0.0,
        1.0,
    ) ** 0.85
    return 255.0 - 255.0 * edge_weight * strength


def adaptive_edge_threshold(
    edges: np.ndarray,
    detail_level: float = 1.0,
) -> float:
    """Ước lượng ngưỡng Sobel từ phân bố biên của chính ảnh đầu vào."""
    positive_edges = edges[edges > 0]
    if positive_edges.size == 0:
        return 255.0

    percentile = float(np.clip(65.0 / detail_level, 40.0, 85.0))
    threshold = float(np.percentile(positive_edges, percentile))
    return float(np.clip(threshold, 15.0, 80.0))


def _validate(
    image: np.ndarray,
    smoothing: str,
    filter_size: int,
    dodge_sigma: float,
    edge_threshold: float,
    edge_weight: float,
) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image phải là NumPy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image phải là ảnh RGB uint8")
    if smoothing not in ("gaussian", "bilateral"):
        raise ValueError("smoothing phải là 'gaussian' hoặc 'bilateral'")
    if not isinstance(filter_size, (int, np.integer)) or isinstance(filter_size, bool):
        raise TypeError("filter_size phải là số nguyên")
    if filter_size <= 0 or filter_size % 2 == 0:
        raise ValueError("filter_size phải là số lẻ dương")
    if not np.isfinite(dodge_sigma) or dodge_sigma <= 0:
        raise ValueError("dodge_sigma phải là số hữu hạn dương")
    if not np.isfinite(edge_threshold) or not 0 <= edge_threshold <= 255:
        raise ValueError("edge_threshold phải nằm trong khoảng 0–255")
    if not np.isfinite(edge_weight) or not 0 <= edge_weight <= 1:
        raise ValueError("edge_weight phải nằm trong khoảng 0–1")


def generate_sketch_steps(
    image: np.ndarray,
    smoothing: str = "bilateral",
    filter_size: int = 5,
    sigma_space: float = 2.0,
    sigma_color: float = 50.0,
    dodge_sigma: float = 8.0,
    edge_threshold: float = 40.0,
    contrast: float = 1.05,
    brightness: float = 0.0,
    detail_level: float = 1.0,
    edge_weight: float = 0.75,
    automatic_threshold: bool = True,
) -> dict[str, np.ndarray]:
    """Chạy pipeline và trả ảnh trung gian cùng ảnh kết quả cuối."""
    _validate(
        image, smoothing, filter_size, dodge_sigma,
        edge_threshold, edge_weight,
    )
    if not np.isfinite(detail_level) or not 0.5 <= detail_level <= 1.5:
        raise ValueError("detail_level phải nằm trong khoảng 0.5–1.5")

    # Ba bước bắt buộc của đề tài: grayscale → smoothing → Sobel.
    gray = to_grayscale(image)
    adjusted = adjust_tone(percentile_stretch(gray), contrast, brightness)
    if smoothing == "gaussian":
        smooth = gaussian_filter(adjusted, size=filter_size)
    elif smoothing == "bilateral":
        smooth = bilateral_filter(
            adjusted, filter_size, sigma_space, sigma_color,
        )
    else:
        smooth = bilateral_filter(
            adjusted, filter_size, sigma_space, sigma_color,
        )

    # Biên chính lấy từ ảnh làm mịn đã chọn. Một ảnh Gaussian nhẹ 3×3
    # bổ sung các chi tiết nhỏ bị Bilateral/kernel lớn làm yếu đi.
    main_edges = sobel_edge_detection(smooth)
    detail_source = gaussian_filter(adjusted, size=3)
    detail_edges = sobel_edge_detection(detail_source)
    edges = np.maximum(
        main_edges,
        np.rint(0.35 * detail_edges).astype(np.uint8),
    )
    dodge = _dodge_blend(smooth, dodge_sigma)
    manual_threshold = float(np.clip(edge_threshold / detail_level, 0, 255))
    if automatic_threshold:
        automatic = adaptive_edge_threshold(edges, detail_level)
        threshold = 0.75 * automatic + 0.25 * manual_threshold
    else:
        threshold = manual_threshold

    if np.ptp(smooth) == 0:
        final = np.full_like(smooth, 255)
    else:
        # Dodge tạo bóng chì, Sobel giữ nét quan trọng và tone xám cục bộ
        # tạo chiều sâu mà không làm nền phẳng bị tối.
        edge_factor = edge_weight
        shade_factor = 0.30
        tone_factor = 0.14
        shade = 255.0 - shade_factor * (255.0 - dodge.astype(np.float32))
        lines = _edge_layer(edges, threshold, edge_factor)
        # Chỉ giữ tone xám rõ ở gần đường nét; vùng phẳng được đẩy về
        # gần màu giấy trắng để tránh biến toàn bộ nền tối thành mảng xám.
        line_strength = np.clip(
            (255.0 - lines) / max(255.0 * edge_factor, 1.0),
            0.0,
            1.0,
        )
        local_tone_factor = tone_factor * (0.25 + 0.75 * line_strength)
        tone = 255.0 - local_tone_factor * (
            255.0 - smooth.astype(np.float32)
        )
        final = np.minimum(np.minimum(shade, tone), lines)

    final = np.clip(np.rint(final), 0, 255).astype(np.uint8)
    return {
        "gray": gray,
        "adjusted": adjusted,
        "smooth": smooth,
        "edges": edges,
        "dodge": dodge,
        "final": final,
    }


def generate_sketch(
    image: np.ndarray,
    smoothing: str = "bilateral",
    filter_size: int = 5,
    sigma_space: float = 2.0,
    sigma_color: float = 50.0,
    dodge_sigma: float = 8.0,
    edge_threshold: float = 40.0,
    contrast: float = 1.05,
    brightness: float = 0.0,
    detail_level: float = 1.0,
    edge_weight: float = 0.75,
    automatic_threshold: bool = True,
) -> np.ndarray:
    """Giữ API cũ và trả riêng ảnh tranh vẽ cuối cùng."""
    return generate_sketch_steps(
        image, smoothing, filter_size, sigma_space, sigma_color,
        dodge_sigma, edge_threshold, contrast, brightness,
        detail_level, edge_weight, automatic_threshold,
    )["final"]
