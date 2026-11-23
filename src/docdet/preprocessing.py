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


def load_mask(source: Image.Image | str | Path) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("L")
    with Image.open(source) as image:
        return image.convert("L")


def resize_image(image: Image.Image) -> Image.Image:
    return image.resize(MODEL_IMAGE_SIZE, Image.Resampling.LANCZOS)


def resize_mask(mask: Image.Image) -> Image.Image:
    return mask.resize(MODEL_IMAGE_SIZE, Image.Resampling.NEAREST)


def preprocess_image(source: Image.Image | str | Path) -> np.ndarray:
    image = resize_image(load_rgb(source))
    return np.asarray(image, dtype=np.float32) / 255.0


def preprocess_mask(source: Image.Image | str | Path) -> np.ndarray:
    mask = resize_mask(load_mask(source))
    return np.asarray(mask, dtype=np.float32) / 255.0


def verify_nonempty_mask(mask: np.ndarray) -> None:
    if not mask.any():
        raise ValueError("preprocessed tamper mask must be non-empty")


def preprocess_sample(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    require_nonempty: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    image = preprocess_image(image_path)
    mask = preprocess_mask(mask_path) if mask_path is not None else None
    if require_nonempty:
        if mask is None:
            raise ValueError("a tamper mask is required")
        verify_nonempty_mask(mask)
    return image, mask


__all__ = [
    "MODEL_IMAGE_SIZE",
    "load_mask",
    "load_rgb",
    "preprocess_image",
    "preprocess_mask",
    "preprocess_sample",
    "resize_image",
    "resize_mask",
    "verify_nonempty_mask",
]
