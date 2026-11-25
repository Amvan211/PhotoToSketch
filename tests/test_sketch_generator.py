
import numpy as np
import pytest

from src.sketch_generator import (
    adaptive_edge_threshold,
    adjust_tone,
    generate_sketch,
    generate_sketch_steps,
    percentile_stretch,
)


def test_generate_sketch_output_shape_and_dtype():
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    result = generate_sketch(image)

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_generate_sketch_uniform_image_is_white():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)

    result = generate_sketch(image)

    assert np.all(result == 255)


@pytest.mark.parametrize(
    "smoothing",
    ["gaussian", "bilateral"],
)
def test_generate_sketch_supports_smoothing_modes(smoothing):
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[:, 6:] = 255

    result = generate_sketch(image, smoothing=smoothing)

    assert result.shape == image.shape[:2]
    assert result.dtype == np.uint8


def test_generate_sketch_rejects_invalid_smoothing():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        generate_sketch(image, smoothing="invalid")


def test_generate_sketch_rejects_grayscale_input():
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        generate_sketch(image)


@pytest.mark.parametrize("filter_size", [0, 2])
def test_generate_sketch_rejects_invalid_filter_size(filter_size):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        generate_sketch(image, filter_size=filter_size)


def test_generate_sketch_rejects_invalid_algorithm_parameters():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        generate_sketch(image, dodge_sigma=0)
    with pytest.raises(ValueError):
        generate_sketch(image, edge_threshold=300)


def test_pencil_uses_complete_pipeline():
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[5:19, 8:16] = 180

    steps = generate_sketch_steps(image)

    assert set(steps) == {
        "gray", "adjusted", "smooth", "edges", "dodge", "final"
    }
    assert all(value.shape == (24, 24) for value in steps.values())
    assert all(value.dtype == np.uint8 for value in steps.values())
    assert np.any(steps["edges"] > 0)


def test_percentile_stretch_and_tone_adjustment():
    image = np.array([[10, 20, 30]], dtype=np.uint8)
    stretched = percentile_stretch(image)
    brighter = adjust_tone(stretched, contrast=1.0, brightness=20)

    assert stretched[0, 0] == 0
    assert stretched[0, -1] == 255
    assert np.all(brighter >= stretched)


def test_new_parameters_are_validated():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        generate_sketch(image, edge_weight=1.5)
    with pytest.raises(ValueError):
        generate_sketch(image, detail_level=2.0)
    with pytest.raises(ValueError):
        generate_sketch(image, contrast=0)


def test_adaptive_threshold_changes_with_edge_distribution():
    weak_edges = np.array([[0, 10, 15, 20, 25]], dtype=np.uint8)
    strong_edges = np.array([[0, 80, 120, 180, 255]], dtype=np.uint8)

    weak_threshold = adaptive_edge_threshold(weak_edges)
    strong_threshold = adaptive_edge_threshold(strong_edges)

    assert 15 <= weak_threshold <= 80
    assert 15 <= strong_threshold <= 80
    assert strong_threshold > weak_threshold


def test_automatic_threshold_can_be_disabled():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[:, 15:] = 100

    automatic = generate_sketch(
        image,
        edge_threshold=80,
        automatic_threshold=True,
    )
    manual = generate_sketch(
        image,
        edge_threshold=80,
        automatic_threshold=False,
    )

    assert automatic.shape == manual.shape
    assert automatic.dtype == manual.dtype == np.uint8


def test_high_contrast_image_is_not_stretched_again():
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    result = percentile_stretch(image)

    assert np.array_equal(result, image)


def test_pencil_sketch_keeps_soft_tones_and_edges():
    image = np.full((40, 40, 3), 180, dtype=np.uint8)
    image[10:30, 15:25] = 50

    steps = generate_sketch_steps(image)
    result = steps["final"]

    assert np.any((result > 0) & (result < 255))
    assert np.any(steps["edges"] > 0)
    assert result.mean() > 150
