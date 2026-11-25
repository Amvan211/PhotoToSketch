
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.image_io import normalize_image, read_image, resize_for_processing, save_image


def test_read_and_save_image(tmp_path: Path):
    # Tạo một ảnh RGB mẫu để kiểm thử.
    original = np.zeros((10, 12, 3), dtype=np.uint8)
    original[:, :, 0] = 255

    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output" / "result.png"

    Image.fromarray(original).save(input_path)

    loaded = read_image(input_path)

    assert loaded.shape == (10, 12, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, original)

    save_image(loaded, output_path)

    assert output_path.exists()
    assert np.array_equal(read_image(output_path), original)


def test_normalize_transparent_image_uses_white_background():
    source = Image.new("RGBA", (2, 2), (255, 0, 0, 0))

    result = normalize_image(source)

    assert result.mode == "RGB"
    assert np.array_equal(np.asarray(result)[0, 0], [255, 255, 255])


def test_resize_for_processing_preserves_aspect_ratio():
    source = Image.new("RGB", (400, 200))

    result = resize_for_processing(source, max_pixels=20_000)

    assert result.width * result.height <= 20_000
    assert result.width / result.height == pytest.approx(2.0, rel=0.02)


def test_resize_for_processing_keeps_small_image_size():
    source = Image.new("RGB", (20, 10))

    result = resize_for_processing(source, max_pixels=1_000)

    assert result.size == source.size
