
import numpy as np
import pytest

from src.edge_detection import (
    sobel_gradients,
    sobel_edge_detection,
)


def test_sobel_preserves_shape_and_dtype():
    image = np.zeros((10, 12), dtype=np.uint8)

    result = sobel_edge_detection(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_sobel_constant_image_has_no_edges():
    image = np.full((10, 10), 100, dtype=np.uint8)

    result = sobel_edge_detection(image)

    assert np.all(result == 0)


def test_sobel_detects_vertical_boundary():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 5:] = 255

    result = sobel_edge_detection(image)

    assert np.max(result) > 0
    assert np.any(result[:, 3:7] > 0)


def test_sobel_gradients_return_two_arrays():
    image = np.zeros((8, 8), dtype=np.uint8)
    image[:, 4:] = 255

    gx, gy = sobel_gradients(image)

    assert gx.shape == image.shape
    assert gy.shape == image.shape
    assert np.issubdtype(gx.dtype, np.floating)
    assert np.issubdtype(gy.dtype, np.floating)


def test_sobel_rejects_invalid_input():
    image = np.zeros((5, 5, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        sobel_edge_detection(image)


def test_sobel_robust_normalization_keeps_weaker_edges_visible():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, 30:] = 60
    image[:, 70:] = 255

    result = sobel_edge_detection(image)

    assert np.any(result[:, 28:32] > 0)
    assert np.any(result[:, 68:72] > 0)
