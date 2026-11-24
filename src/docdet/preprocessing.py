"""Deterministic Pillow preprocessing shared by training and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MODEL_IMAGE_SIZE = (256, 160)


def load_rgb(source: Image.Image | str | Path) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    with Image.open(source) as image:
        return image.convert("RGB")


def resize_image(image: Image.Image) -> Image.Image:
    return image.resize(MODEL_IMAGE_SIZE, Image.Resampling.LANCZOS)


def preprocess_image(source: Image.Image | str | Path) -> np.ndarray:
    image = resize_image(load_rgb(source))
    return np.asarray(image, dtype=np.float32) / 255.0


__all__ = [
    "MODEL_IMAGE_SIZE",
    "load_rgb",
    "preprocess_image",
    "resize_image",
]
