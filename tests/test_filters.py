
import numpy as np
import pytest

from src.filters import (
    convolve2d,
    gaussian_kernel,
    gaussian_filter,
    bilateral_filter
)


def test_convolve_with_identity_kernel():
    image = np.array(
        [[10, 20], [30, 40]],
        dtype=np.uint8,
    )
    kernel = np.array([[1]], dtype=np.float32)

    result = convolve2d(image, kernel)

    assert np.allclose(result, image)


def test_convolve_rejects_even_kernel():
    image = np.zeros((5, 5), dtype=np.uint8)
    kernel = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        convolve2d(image, kernel)


def test_gaussian_kernel_is_normalized():
    kernel = gaussian_kernel(size=5, sigma=1.0)

    assert kernel.shape == (5, 5)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.all(kernel >= 0)


def test_gaussian_kernel_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        gaussian_kernel(size=4, sigma=1.0)

    with pytest.raises(ValueError):
        gaussian_kernel(size=3, sigma=0)


def test_gaussian_filter_preserves_constant_image():
    image = np.full((10, 10), 100, dtype=np.uint8)

    result = gaussian_filter(image, size=3, sigma=1.0)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.allclose(result, 100)


def test_gaussian_filter_reduces_impulse():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[4, 4] = 255

    result = gaussian_filter(image, size=3, sigma=1.0)

    assert result[4, 4] < 255
    assert result[4, 4] > 0


def test_bilateral_filter_preserves_constant_image():
    image = np.full((7, 7), 120, dtype=np.uint8)

    result = bilateral_filter(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all(result == 120)


def test_bilateral_filter_preserves_single_pixel_value():
    image = np.zeros((7, 7), dtype=np.uint8)
    image[3, 3] = 255

    result = bilateral_filter(
        image,
        size=3,
        sigma_space=2.0,
        sigma_color=10.0,
    )

    # Giá trị tại tâm không bị các pixel tối làm thay đổi đáng kể.
    assert result[3, 3] == 255
    assert result.shape == image.shape


def test_bilateral_filter_rejects_even_size():
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(image, size=2)


def test_bilateral_filter_rejects_invalid_sigma():
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        bilateral_filter(image, sigma_space=0)

    with pytest.raises(ValueError):
        bilateral_filter(image, sigma_color=-1)
