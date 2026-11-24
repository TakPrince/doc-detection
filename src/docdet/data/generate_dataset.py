"""Deterministic matched-pair dataset generation for the MVP."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docdet.config import load_dataset_config, resolve_project_path
from docdet.data.identities import load_identities
from docdet.data.render import render_document
from docdet.data.tamper import (
    TAMPER_CATEGORIES,
    TARGET_FIELDS,
    TamperResult,
    null_edit,
    tamper_specimen,
)
from docdet.seeding import derive_seed

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    identities: Sequence[Mapping[str, Any]]
    splits: Mapping[str, str]
    output_dir: Path
    manifest_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("generation seed must be an integer")
        object.__setattr__(
            self, "identities", tuple(dict(record) for record in self.identities)
        )
        object.__setattr__(self, "splits", dict(self.splits))
        object.__setattr__(self, "output_dir", resolve_project_path(self.output_dir))
        object.__setattr__(
            self, "manifest_path", resolve_project_path(self.manifest_path)
        )


@dataclass(frozen=True)
class GenerationResult:
    manifest: list[dict[str, Any]]
    output_dir: Path
    manifest_path: Path


def load_split_map(path: str | Path) -> dict[str, str]:
    split_path = resolve_project_path(path)
    with split_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_map: dict[str, str] = {}
    for row in rows:
        identity_id = row["identity_id"]
        if identity_id in split_map:
            raise ValueError(f"Duplicate identity in split file: {identity_id}")
        split_map[identity_id] = row["split"]
    return split_map


def dataset_statistics(manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = ("GENUINE", "TAMPERED")
    templates = tuple(sorted({row["template_id"] for row in manifest}))
    splits = tuple(sorted({row["split"] for row in manifest}))
    return {
        "image_count": len(manifest),
        "identity_count": len({row["identity_id"] for row in manifest}),
        "pair_count": len({row["pair_id"] for row in manifest}),
        "label_counts": {
            label: sum(row["label"] == label for row in manifest) for label in labels
        },
        "category_counts": {
            category: sum(row["category"] == category for row in manifest)
            for category in TAMPER_CATEGORIES
        },
        "template_counts": {
            template: sum(row["template_id"] == template for row in manifest)
            for template in templates
        },
        "split_counts": {
            split: sum(row["split"] == split for row in manifest) for split in splits
        },
    }


def _manifest_entry(
    config: GenerationConfig,
    record: Mapping[str, Any],
    pair_id: str,
    image_id: str,
    label: str,
    category: str,
    result: TamperResult,
) -> dict[str, Any]:
    split = config.splits[record["identity_id"]]
    return {
        "schema_version": 1,
        "image_id": image_id,
        "pair_id": pair_id,
        "group_id": record["group_id"],
        "identity_id": record["identity_id"],
        "template_id": record["template_id"],
        "template_family": record["template_family"],
        "split": split,
        "label": label,
        "category": category,
        "target_field": TARGET_FIELDS[category],
        "image_path": str(Path("images") / split / f"{image_id}.png"),
        "operation": result.metadata["operation"],
        "identifier": record["identifier"],
        "specimen_label": record["specimen_label"],
    }


def generate_dataset(config: GenerationConfig) -> GenerationResult:
    output_dir = config.output_dir
    manifest_path = config.manifest_path
    for split in SPLIT_NAMES:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []

    for identity_index, record in enumerate(config.identities):
        identity_id = record["identity_id"]
        if identity_id not in config.splits:
            raise ValueError(f"Missing split assignment for identity: {identity_id}")

        category = TAMPER_CATEGORIES[identity_index % len(TAMPER_CATEGORIES)]
        pair_id = f"{identity_id}__{category}"
        pair_seed = derive_seed(config.seed, "pair", identity_id, category)
        tamper_seed = derive_seed(config.seed, "tamper", identity_id, category)

        specimen = render_document(record)
        genuine = null_edit(specimen, record, category, pair_seed)
        tampered = tamper_specimen(specimen, record, category, tamper_seed)

        for label, result in (("GENUINE", genuine), ("TAMPERED", tampered)):
            image_id = f"{pair_id}__{label}"
            image_path = output_dir / "images" / config.splits[identity_id] / f"{image_id}.png"
            result.image.save(image_path, format="PNG")
            manifest.append(
                _manifest_entry(
                    config, record, pair_id, image_id, label, category, result
                )
            )

    manifest_path.write_text(
        "".join(
            json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
            for entry in manifest
        ),
        encoding="utf-8",
    )
    return GenerationResult(
        manifest=manifest,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic matched-pair dataset"
    )
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--identities", default="data/identities/identities.json")
    parser.add_argument("--splits", default="data/splits/splits_A.csv")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--manifest", default="data/manifests/dataset.jsonl")
    args = parser.parse_args(argv)

    records = load_identities(args.identities)
    dataset_config = load_dataset_config(args.config)
    config = GenerationConfig(
        seed=dataset_config.seed,
        identities=records,
        splits=load_split_map(args.splits),
        output_dir=Path(args.output_dir),
        manifest_path=Path(args.manifest),
    )
    result = generate_dataset(config)
    stats = dataset_statistics(result.manifest)
    print(
        f"Wrote {stats['image_count']} images to {result.output_dir} "
        f"(identities={stats['identity_count']}, pairs={stats['pair_count']})"
    )
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "dataset_statistics",
    "generate_dataset",
    "load_split_map",
]
