
import numpy as np
import pytest

from src.grayscale import to_grayscale


def test_grayscale_shape_and_dtype():
    image = np.zeros((10, 12, 3), dtype=np.uint8)

    gray = to_grayscale(image)

    assert gray.shape == (10, 12)
    assert gray.dtype == np.uint8


def test_grayscale_known_colors():
    image = np.array(
        [[
            [255, 0, 0],      # Đỏ
            [0, 255, 0],      # Xanh lá
            [0, 0, 255],      # Xanh dương
            [255, 255, 255],  # Trắng
            [0, 0, 0],        # Đen
        ]],
        dtype=np.uint8,
    )

    gray = to_grayscale(image)

    expected = np.array(
        [[76, 150, 29, 255, 0]],
        dtype=np.uint8,
    )

    assert np.array_equal(gray, expected)


def test_grayscale_rejects_invalid_input():
    image = np.zeros((10, 12), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_grayscale(image)
