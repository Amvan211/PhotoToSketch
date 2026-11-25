"""Giao diện Streamlit cho ứng dụng PhotoToSketch."""

import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.image_io import normalize_image, resize_for_processing
from src.sketch_generator import generate_sketch_steps


MAX_PROCESSING_PIXELS = 1_500_000
DEFAULTS = {
    "smoothing": "bilateral",
    "smooth_level": "balanced",
    "contrast": 1.05,
    "brightness": 0,
    "detail": 1.0,
    "edge_weight": 0.75,
    "stroke": "balanced",
    "custom_threshold": False,
    "sobel_threshold": 40,
    "auto_threshold": True,
}
SMOOTHING_PRESETS = {
    "detail": (3, 1.5, 30.0),
    "balanced": (5, 2.0, 50.0),
    "strong": (7, 3.0, 75.0),
}


def reset_controls() -> None:
    """Khôi phục các widget về bộ tham số thực nghiệm mặc định."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


@st.cache_data(show_spinner=False, max_entries=8)
def process_cached(image: np.ndarray, parameters: tuple) -> dict[str, np.ndarray]:
    """Cache kết quả theo nội dung ảnh và toàn bộ tham số xử lý."""
    (
        smoothing, filter_size, sigma_space, sigma_color,
        dodge_sigma, edge_threshold, contrast, brightness,
        detail_level, edge_weight,
        automatic_threshold,
    ) = parameters
    return generate_sketch_steps(
        image=image,
        smoothing=smoothing,
        filter_size=filter_size,
        sigma_space=sigma_space,
        sigma_color=sigma_color,
        dodge_sigma=dodge_sigma,
        edge_threshold=edge_threshold,
        contrast=contrast,
        brightness=brightness,
        detail_level=detail_level,
        edge_weight=edge_weight,
        automatic_threshold=automatic_threshold,
    )


st.set_page_config(page_title="PhotoToSketch", page_icon="🎨", layout="wide")
st.title("🎨 PhotoToSketch")
st.caption("Mức xám → làm mịn Gaussian/Bilateral → biên Sobel → tranh vẽ")

for default_key, default_value in DEFAULTS.items():
    st.session_state.setdefault(default_key, default_value)

st.sidebar.header("Cấu hình tranh vẽ")
st.sidebar.info("✏️ **Pencil Sketch** – tranh chì có nét và sắc độ")
smoothing = st.sidebar.selectbox(
    "1. Bộ lọc làm mịn",
    ["bilateral", "gaussian"],
    format_func=lambda value: {
        "bilateral": "Bilateral – giữ biên tốt",
        "gaussian": "Gaussian – làm mịn đều",
    }[value],
    key="smoothing",
)

smooth_level = st.sidebar.radio(
    "2. Mức làm mịn",
    ["detail", "balanced", "strong"],
    format_func=lambda value: {
        "detail": "Nhẹ",
        "balanced": "Vừa",
        "strong": "Mạnh",
    }[value],
    horizontal=True,
    key="smooth_level",
)
filter_size, sigma_space, sigma_color = SMOOTHING_PRESETS[smooth_level]

stroke = st.sidebar.radio(
    "3. Độ đậm nét chì",
    ["light", "balanced", "bold"],
    format_func=lambda value: {"light": "Nhẹ", "balanced": "Vừa", "bold": "Đậm"}[value],
    horizontal=True,
    key="stroke",
)
dodge_sigma = {"light": 11.0, "balanced": 8.0, "bold": 6.0}[stroke]
edge_threshold = {"light": 60.0, "balanced": 40.0, "bold": 22.0}[stroke]

with st.sidebar.expander("⚙️ Điều chỉnh chất lượng", expanded=False):
    contrast = st.slider(
        "Độ tương phản", 0.70, 1.40, step=0.05, key="contrast",
        help="Tăng để tách rõ vùng sáng và tối trước khi tìm biên.",
    )
    brightness = st.slider(
        "Độ sáng", -50, 50, step=5, key="brightness",
    )
    detail_level = st.slider(
        "Mức giữ chi tiết", 0.50, 1.50, step=0.05, key="detail",
        help="Cao hơn giữ nhiều biên yếu hơn nhưng có thể tăng nhiễu.",
    )
    edge_weight = st.slider(
        "Trọng số đường biên", 0.0, 1.0, step=0.05, key="edge_weight",
        help="Mức đóng góp của Sobel vào đường nét bút chì.",
    )
    automatic_threshold = st.toggle(
        "Tự cân chỉnh ngưỡng theo ảnh",
        key="auto_threshold",
        help="Dùng thống kê Sobel của ảnh để hạn chế phải chỉnh từng ảnh.",
    )
    custom_threshold = st.toggle(
        "Tự đặt ngưỡng Sobel",
        key="custom_threshold",
        help="Tắt để ngưỡng tự đi theo lựa chọn Độ đậm nét.",
    )
    if custom_threshold:
        edge_threshold = float(st.slider(
            "Ngưỡng Sobel",
            5,
            100,
            step=5,
            key="sobel_threshold",
            help="Thấp: nhiều nét hơn. Cao: ít nhiễu nền hơn.",
        ))
    else:
        st.caption(f"Ngưỡng Sobel hiện tại: **{edge_threshold:.0f}**")
    if automatic_threshold:
        st.caption(
            "Kết quả dùng 75% ngưỡng tự động và 25% ngưỡng ở trên."
        )
    if smoothing == "bilateral":
        st.caption(
            f"Bilateral: kernel {filter_size}×{filter_size}, "
            f"σ không gian {sigma_space}, σ cường độ {sigma_color}."
        )

with st.sidebar.expander("📘 Chọn tham số theo loại ảnh"):
    st.markdown(
        """
Ngưỡng Sobel **thấp** giữ nhiều nét; ngưỡng **cao** loại nhiều biên yếu và
nhiễu nền. Hãy bắt đầu từ gợi ý dưới đây rồi chỉ chỉnh nếu thật sự cần.

| Loại ảnh | Bộ lọc | Độ đậm | Ngưỡng | Chi tiết |
|---|---|---:|---:|---:|
| Chân dung | Bilateral vừa | Vừa | 35–45 | 0.9–1.1 |
| Anime/minh họa | Bilateral nhẹ | Nhẹ | 50–65 | 0.8–1.0 |
| Thiên nhiên | Bilateral mạnh | Vừa | 40–55 | 0.8–1.0 |
| Y tế | Bilateral nhẹ | Vừa | 25–40 | 1.1–1.3 |
| Công nghiệp | Gaussian/Bilateral | Đậm | 20–35 | 1.1–1.4 |

**Dấu hiệu cần chỉnh:**

- Mất mắt, tóc hoặc cạnh nhỏ → giảm ngưỡng hoặc tăng giữ chi tiết.
- Quá nhiều nét nền → tăng ngưỡng hoặc tăng làm mịn.
- Nét quá đen → giảm độ đậm hoặc giảm trọng số đường biên.
- Ảnh phẳng, thiếu chiều sâu → tăng tương phản nhẹ.
        """
    )

st.sidebar.button(
    "↺ Khôi phục mặc định",
    on_click=reset_controls,
    use_container_width=True,
)

parameters = (
    smoothing, filter_size, sigma_space, sigma_color,
    dodge_sigma, edge_threshold, contrast, brightness,
    detail_level, edge_weight,
    automatic_threshold,
)

uploaded_file = st.file_uploader(
    "Chọn ảnh đầu vào",
    help="Hỗ trợ PNG, JPEG, WebP, BMP, TIFF và các định dạng Pillow đọc được.",
)

if uploaded_file is None:
    st.info("Hãy tải một ảnh lên để bắt đầu.")
    st.stop()
    raise SystemExit  # Chỉ có tác dụng khi chạy trực tiếp ngoài Streamlit.

file_bytes = uploaded_file.getvalue()
try:
    with Image.open(io.BytesIO(file_bytes)) as source:
        original_image = normalize_image(source)
    processing_image = resize_for_processing(original_image, MAX_PROCESSING_PIXELS)
    original_array = np.asarray(processing_image, dtype=np.uint8)
except (UnidentifiedImageError, OSError, ValueError) as error:
    st.error(f"Tệp không phải ảnh hợp lệ hoặc không được hỗ trợ: {error}")
    st.stop()

if processing_image.size != original_image.size:
    st.info(
        f"Ảnh được thu nhỏ từ {original_image.width}×{original_image.height} "
        f"xuống {processing_image.width}×{processing_image.height} để xử lý ổn định."
    )

request_key = (hashlib.sha256(file_bytes).hexdigest(), parameters)
if st.sidebar.button("▶️ Chuyển thành tranh vẽ", type="primary", use_container_width=True):
    with st.spinner("Đang xử lý ảnh..."):
        try:
            st.session_state["steps"] = process_cached(original_array, parameters)
            st.session_state["result_key"] = request_key
        except (TypeError, ValueError) as error:
            st.error(f"Không thể xử lý ảnh: {error}")
            st.session_state.pop("result_key", None)

left, right = st.columns(2)
with left:
    st.subheader("Ảnh gốc")
    st.image(original_image, use_container_width=True)
with right:
    st.subheader("Ảnh tranh vẽ")
    if st.session_state.get("result_key") == request_key:
        steps = st.session_state["steps"]
        st.image(steps["final"], use_container_width=True, clamp=True)
        buffer = io.BytesIO()
        Image.fromarray(steps["final"]).save(buffer, "PNG")
        st.download_button(
            "⬇️ Tải kết quả PNG",
            buffer.getvalue(),
            file_name="pencil_sketch.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.info("Nhấn **Chuyển thành tranh vẽ** để tạo kết quả.")

if st.session_state.get("result_key") == request_key:
    with st.expander("🔬 Xem các bước xử lý trung gian"):
        labels = [
            ("gray", "1. Ảnh mức xám"),
            ("adjusted", "2. Điều chỉnh tương phản"),
            ("smooth", "3. Ảnh làm mịn"),
            ("edges", "4. Biên Sobel"),
            ("dodge", "5. Dodge Blend"),
            ("final", "6. Kết quả cuối"),
        ]
        columns = st.columns(3)
        for index, (key, label) in enumerate(labels):
            with columns[index % 3]:
                st.caption(label)
                st.image(st.session_state["steps"][key], use_container_width=True)
