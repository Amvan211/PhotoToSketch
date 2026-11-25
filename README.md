# PhotoToSketch

Ứng dụng chuyển ảnh thành tranh vẽ cho bài tập lớn môn Xử lý ảnh số – Đề
tài 4. Dự án tự cài đặt các thuật toán bằng NumPy, dùng Pillow để đọc/lưu
ảnh và Streamlit làm giao diện. Dự án **không dùng OpenCV, AI hoặc API ngoài**.

## Chức năng

- Tải PNG, JPEG, WebP, BMP, TIFF và các định dạng Pillow hỗ trợ.
- Chuẩn hóa RGB/RGBA, xoay theo EXIF và giảm kích thước ảnh quá lớn.
- Chuyển RGB sang mức xám theo trọng số độ chói.
- Làm mịn bằng Gaussian Filter hoặc Bilateral Filter tự cài đặt.
- Phát hiện biên bằng Sobel, chuẩn hóa bền vững theo percentile 99.
- Giao diện tập trung vào Pencil Sketch có đường nét và sắc độ tự nhiên.
- Điều chỉnh độ sáng, tương phản, chi tiết, làm mịn và trọng số biên.
- Tự cân chỉnh ngưỡng Sobel theo phân bố biên của từng ảnh.
- Xem sáu bước xử lý trung gian và tải kết quả PNG.
- Cache kết quả theo ảnh và tham số để tránh tính toán lặp lại.

## Pipeline xử lý

```text
Ảnh RGB
  → ảnh xám
  → kéo giãn tương phản percentile 1–99
  → điều chỉnh sáng/tương phản
  → Gaussian hoặc Bilateral
  → gradient Sobel
  → Dodge Blend bổ trợ sắc độ
  → phối hợp theo phong cách
  → ảnh tranh vẽ
```

Ba bước bắt buộc **mức xám → làm mịn → phát hiện biên Sobel** luôn có trong
pipeline chính. Dodge Blend không thay thế Sobel.

## Công thức chính

Chuyển mức xám:

```text
Gray = 0.299R + 0.587G + 0.114B
```

Gradient Sobel:

```text
G = sqrt(Gx² + Gy²)
```

Color Dodge, với mẫu số được chặn tối thiểu bằng 1:

```text
Sketch = Gray × 255 / max(255 - Blur(255 - Gray), 1)
```

Bilateral Filter dùng tích hai trọng số:

```text
w(p,q) = exp(-||p-q||² / 2σs²) × exp(-(I(p)-I(q))² / 2σr²)
```

## Chế độ sản phẩm

**Pencil Sketch** là chế độ duy nhất của ứng dụng: Dodge tạo bóng chì,
Sobel giữ đường nét quan trọng và sắc độ xám nhẹ tạo chiều sâu.

## Ý nghĩa tham số

- **Mức làm mịn:** chọn kernel và sigma theo ba preset. Mức cao giảm nhiễu
  tốt hơn nhưng có thể làm mất chi tiết nhỏ.
- **Độ đậm nét:** đồng thời điều chỉnh ngưỡng Sobel và sigma Dodge.
- **Độ tương phản:** tách rõ vùng sáng/tối trước khi phát hiện biên.
- **Độ sáng:** cộng một giá trị vào ảnh xám đã chuẩn hóa.
- **Mức giữ chi tiết:** tăng giá trị sẽ hạ ngưỡng biên hiệu dụng.
- **Tự cân chỉnh ngưỡng:** kết hợp 75% ngưỡng thích nghi theo percentile
  biên của ảnh với 25% ngưỡng do người dùng chọn.
- **Trọng số đường biên:** mức đóng góp của Sobel trong Pencil Sketch.
- **Sigma không gian:** phạm vi lân cận của Bilateral Filter.
- **Sigma cường độ:** mức chênh lệch sáng tối được phép trộn.

Bộ tham số mặc định sau thực nghiệm:

```text
Phong cách       Pencil Sketch
Làm mịn          Bilateral, kernel 5×5
Sigma không gian 2.0
Sigma cường độ   50.0
Ngưỡng Sobel     40
Sigma Dodge      8.0
Tương phản       1.05
Độ sáng          0
Giữ chi tiết     1.0
Trọng số biên    0.75
Ngưỡng tự động   Bật
```

## Cài đặt và chạy

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Kiểm thử

```powershell
python -m pytest -v
```

Test bao phủ I/O, grayscale, convolution, Gaussian, Bilateral, Sobel,
Pencil Sketch, ảnh trung gian và kiểm tra tham số.

## Thực nghiệm

Chạy toàn bộ ảnh trong `data/input/` và `data/test_cases/`:

```powershell
python evaluate.py
```

Kết quả được lưu tại:

- `results/output/`: kết quả Pencil với Gaussian và Bilateral.
- `results/comparison/`: ảnh ghép so sánh trực quan.
- `results/benchmark.csv`: kích thước ảnh, tham số và thời gian thực tế.

Không dùng PSNR/SSIM làm tiêu chí duy nhất vì tranh vẽ không có mục tiêu tái
tạo ảnh gốc. Nên đánh giá thêm độ rõ nét, nhiễu nền, sắc độ, chi tiết và cảm
nhận giống tranh bút chì.

## Hạn chế

- Chất lượng phụ thuộc độ tương phản, nhiễu và độ phức tạp của nền.
- Ảnh có nền nhiều texture vẫn có thể tạo thêm các nét chì không mong muốn.
- Bilateral tự cài đặt chậm hơn thư viện xử lý ảnh đã tối ưu trên ảnh lớn;
  giao diện tự giảm ảnh trên 1,5 triệu pixel để đảm bảo ổn định.
- DICOM và RAW chuyên dụng chưa được hỗ trợ trực tiếp; nên chuyển sang PNG
  hoặc TIFF trước khi sử dụng.
- `data/test_cases/` hiện chưa có ảnh y tế hoặc công nghiệp. Cần bổ sung để
  kết luận thực nghiệm đầy đủ cho các nhóm ảnh giảng viên yêu cầu.
