"""Chạy thực nghiệm lặp lại được trên toàn bộ ảnh trong data/."""

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.image_io import read_image
from src.edge_detection import sobel_edge_detection
from src.grayscale import to_grayscale
from src.sketch_generator import generate_sketch


INPUT_DIRS = (Path("data/input"), Path("data/test_cases"))
OUTPUT_DIR = Path("results/output")
COMPARISON_DIR = Path("results/comparison")
FILTERS = ("gaussian", "bilateral")


def _panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.contain(image.convert("RGB"), size)
    canvas = Image.new("RGB", (size[0], size[1] + 34), "white")
    canvas.paste(image, ((size[0] - image.width) // 2, 34))
    ImageDraw.Draw(canvas).text((8, 9), label, fill="black")
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [
        path for directory in INPUT_DIRS if directory.exists()
        for path in directory.iterdir() if path.is_file()
    ]
    rows: list[dict[str, object]] = []

    for path in image_paths:
        source = read_image(path)
        baseline_edges = sobel_edge_detection(to_grayscale(source))
        baseline = np.where(baseline_edges > 30, 0, 255).astype(np.uint8)
        generated: dict[str, np.ndarray] = {}
        for smoothing in FILTERS:
            started = time.perf_counter()
            result = generate_sketch(source, smoothing=smoothing)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            generated[smoothing] = result
            output_name = f"{path.stem}_{smoothing}_pencil.png"
            Image.fromarray(result).save(OUTPUT_DIR / output_name)
            rows.append({
                "image": path.name,
                "width": source.shape[1],
                "height": source.shape[0],
                "smoothing": smoothing,
                "filter_size": 5,
                "edge_threshold": 40,
                "edge_weight": 0.75,
                "automatic_threshold": True,
                "elapsed_ms": round(elapsed_ms, 3),
            })

        panels = [
            _panel(Image.fromarray(source), "Original", (360, 360)),
            _panel(Image.fromarray(baseline), "Baseline Sobel", (360, 360)),
        ]
        for smoothing, label in (
            ("gaussian", "Pencil + Gaussian"),
            ("bilateral", "Pencil + Bilateral"),
        ):
            panels.append(
                _panel(Image.fromarray(generated[smoothing]), label, (360, 360))
            )
        comparison = Image.new("RGB", (360 * len(panels), 394), "white")
        for index, panel in enumerate(panels):
            comparison.paste(panel, (index * 360, 0))
        comparison.save(COMPARISON_DIR / f"{path.stem}_comparison.png")

    with (Path("results") / "benchmark.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Evaluated {len(image_paths)} images and created {len(rows)} results.")


if __name__ == "__main__":
    main()
