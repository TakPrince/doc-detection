"""Lazy, manifest-backed dataset loading with an optional tf.data adapter."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from docdet.config import resolve_project_path
from docdet.preprocessing import MODEL_IMAGE_SIZE, preprocess_image

DEFAULT_MANIFEST_PATH = "data/manifests/dataset.jsonl"
DEFAULT_DATA_ROOT = "data/generated"
VALID_SPLITS = ("train", "validation", "test")
LABEL_TO_INT = {"GENUINE": 0, "TAMPERED": 1}


@dataclass(frozen=True)
class Sample:
    image: np.ndarray
    label: int
    metadata: dict[str, Any]


class SyntheticDataset:
    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split: str | None = None,
    ) -> None:
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.manifest_path = resolve_project_path(manifest_path)
        self.data_root = resolve_project_path(data_root)
        self.split = split
        self._image_cache: dict[str, np.ndarray] = {}

    def _load_image(self, image_path: Path) -> np.ndarray:
        key = str(image_path)
        cached = self._image_cache.get(key)
        if cached is None:
            scaled = np.rint(np.asarray(preprocess_image(image_path)) * 255.0)
            cached = scaled.astype(np.uint8)
            self._image_cache[key] = cached
        return cached.astype(np.float32) / 255.0

    def _iter_entries(self) -> Iterator[dict[str, Any]]:
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if self.split is None or entry["split"] == self.split:
                    yield entry

    def __len__(self) -> int:
        return sum(1 for _ in self._iter_entries())

    def __iter__(self) -> Iterator[Sample]:
        for entry in self._iter_entries():
            label_text = entry["label"]
            if label_text not in LABEL_TO_INT:
                raise ValueError(f"Unknown label: {label_text}")
            yield Sample(
                image=self._load_image(self.data_root / entry["image_path"]),
                label=LABEL_TO_INT[label_text],
                metadata=dict(entry),
            )

    def batches(self, batch_size: int) -> Iterator[list[Sample]]:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise ValueError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batch: list[Sample] = []
        for sample in self:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def to_tf_dataset(
    dataset: SyntheticDataset,
    batch_size: int = 32,
    *,
    repeat: bool = False,
):
    if not tensorflow_available():
        raise ImportError("TensorFlow is unavailable; install it separately to use tf.data")
    import tensorflow as tf

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    def generator():
        for sample in dataset:
            yield sample.image, sample.label

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0], 3), tf.float32),
            tf.TensorSpec((), tf.int64),
        ),
    ).batch(batch_size)
    if repeat:
        tf_dataset = tf_dataset.repeat()
    return tf_dataset


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MANIFEST_PATH",
    "LABEL_TO_INT",
    "VALID_SPLITS",
    "Sample",
    "SyntheticDataset",
    "tensorflow_available",
    "to_tf_dataset",
]
