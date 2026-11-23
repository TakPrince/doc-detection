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
    preprocess_mask,
    preprocess_sample,
    resize_image,
)


def _write_image(tmp_path: Path, name: str, color=(10, 20, 30)) -> Path:
    path = tmp_path / name
    Image.new("RGB", (1000, 625), color).save(path)
    return path


def _write_mask(tmp_path: Path, name: str, value: int) -> Path:
    path = tmp_path / name
    mask = Image.new("L", (1000, 625), 0)
    if value:
        mask.paste(255, (40, 40, 200, 160))
    mask.save(path)
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


def test_preprocess_mask_uses_nearest_values_and_expected_shape(tmp_path):
    path = _write_mask(tmp_path, "mask.png", 1)

    result = preprocess_mask(path)

    assert result.shape == (160, 256)
    assert result.dtype == np.float32
    assert set(np.unique(result)) <= {0.0, 1.0}
    assert result.any()


def test_empty_mask_remains_empty_after_preprocessing(tmp_path):
    path = _write_mask(tmp_path, "mask.png", 0)

    assert not preprocess_mask(path).any()


def test_preprocess_sample_rejects_empty_required_mask(tmp_path):
    image_path = _write_image(tmp_path, "image.png")
    mask_path = _write_mask(tmp_path, "mask.png", 0)

    with pytest.raises(ValueError, match="non-empty"):
        preprocess_sample(image_path, mask_path, require_nonempty=True)


def test_preprocess_sample_returns_image_and_mask(tmp_path):
    image_path = _write_image(tmp_path, "image.png")
    mask_path = _write_mask(tmp_path, "mask.png", 1)

    image, mask = preprocess_sample(image_path, mask_path, require_nonempty=True)

    assert image.shape == (160, 256, 3)
    assert mask.shape == (160, 256)
    assert mask.any()
