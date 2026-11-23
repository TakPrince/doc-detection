from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.dataset import SyntheticDataset, tensorflow_available, to_tf_dataset  # noqa: E402

IMAGE_SIZE = (100, 64)


def _write_dataset(tmp_path: Path):
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "manifest.jsonl"
    rows = []
    splits = ("train", "validation", "test")
    for index, split in enumerate(splits):
        identity_id = f"identity-{index}"
        pair_id = f"{identity_id}__TEXT__EASY"
        for label_index, label in enumerate(("GENUINE", "TAMPERED")):
            image_id = f"{pair_id}__{label}"
            image_path = Path("images") / split / f"{image_id}.png"
            mask_path = Path("masks") / split / f"{image_id}__mask.png"
            (data_root / image_path).parent.mkdir(parents=True, exist_ok=True)
            (data_root / mask_path).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", IMAGE_SIZE, (index * 30, 20, 40)).save(data_root / image_path)
            mask = Image.new("L", IMAGE_SIZE, 0)
            if label == "TAMPERED":
                mask.paste(255, (0, 0, 20, 20))
            mask.save(data_root / mask_path)
            rows.append(
                {
                    "image_id": image_id,
                    "pair_id": pair_id,
                    "identity_id": identity_id,
                    "group_id": f"grp-{identity_id}",
                    "template_id": "AAD-1",
                    "template_family": "AAD",
                    "split": split,
                    "label": label,
                    "category": "TEXT",
                    "difficulty": "EASY",
                    "target_field": "full_name",
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "identifier": "100000000001",
                }
            )
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path, data_root, rows


def test_dataset_filters_split_and_returns_labels_metadata(tmp_path):
    manifest_path, data_root, _ = _write_dataset(tmp_path)

    dataset = SyntheticDataset(manifest_path, data_root, split="validation")
    samples = list(dataset)

    assert len(dataset) == 2
    assert len(samples) == 2
    assert [sample.label for sample in samples] == [0, 1]
    assert all(sample.metadata["split"] == "validation" for sample in samples)
    assert all(sample.metadata["image_id"] for sample in samples)
    assert samples[0].image.shape == (160, 256, 3)
    assert samples[0].mask.shape == (160, 256)


def test_dataset_iteration_is_deterministic(tmp_path):
    manifest_path, data_root, _ = _write_dataset(tmp_path)
    dataset = SyntheticDataset(manifest_path, data_root)

    first = [sample.metadata["image_id"] for sample in dataset]
    second = [sample.metadata["image_id"] for sample in dataset]
    assert first == second

    first_images = [sample.image for sample in dataset]
    second_images = [sample.image for sample in dataset]
    assert all(np.array_equal(left, right) for left, right in zip(first_images, second_images))


def test_batches_are_bounded_and_preserve_order(tmp_path):
    manifest_path, data_root, rows = _write_dataset(tmp_path)
    dataset = SyntheticDataset(manifest_path, data_root)

    batches = list(dataset.batches(batch_size=4))

    assert [len(batch) for batch in batches] == [4, 2]
    ids = [sample.metadata["image_id"] for batch in batches for sample in batch]
    assert ids == [row["image_id"] for row in rows]


def test_dataset_rejects_empty_tampered_mask(tmp_path):
    manifest_path, data_root, rows = _write_dataset(tmp_path)
    mask_path = data_root / rows[-1]["mask_path"]
    Image.new("L", IMAGE_SIZE, 0).save(mask_path)
    dataset = SyntheticDataset(manifest_path, data_root)

    with pytest.raises(ValueError, match="non-empty"):
        list(dataset)


def test_dataset_rejects_unknown_split(tmp_path):
    manifest_path, data_root, _ = _write_dataset(tmp_path)

    with pytest.raises(ValueError, match="split"):
        SyntheticDataset(manifest_path, data_root, split="invalid")


def test_tf_adapter_is_unavailable_without_tensorflow(tmp_path):
    manifest_path, data_root, _ = _write_dataset(tmp_path)
    dataset = SyntheticDataset(manifest_path, data_root, split="train")

    if not tensorflow_available():
        with pytest.raises(ImportError, match="TensorFlow"):
            to_tf_dataset(dataset, batch_size=2)
