from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.preprocessing import (  # noqa: E402
    MODEL_IMAGE_SIZE,
    load_rgb,
    preprocess_image,
    resize_image,
)


def _write_image(tmp_path: Path, name: str, color=(10, 20, 30)) -> Path:
    path = tmp_path / name
    Image.new("RGB", (1000, 625), color).save(path)
    return path


def test_preprocess_image_shapes_normalizes_and_converts_to_rgb(tmp_path):
    path = _write_image(tmp_path, "image.png")

    result = preprocess_image(path)

    assert result.shape == (160, 256, 3)
    assert result.dtype == np.float32
    assert 0.0 <= result.min() <= result.max() <= 1.0
    assert np.allclose(result[0, 0], np.array([10, 20, 30], dtype=np.float32) / 255)


def test_preprocess_is_deterministic(tmp_path):
    path = _write_image(tmp_path, "image.png")

    assert np.array_equal(preprocess_image(path), preprocess_image(path))


def test_load_rgb_and_resize_use_expected_sizes(tmp_path):
    path = _write_image(tmp_path, "image.png")
    with Image.open(path) as image:
        rgb = load_rgb(image)
    resized = resize_image(rgb)

    assert rgb.mode == "RGB"
    assert resized.size == MODEL_IMAGE_SIZE
