"""Deterministic Phase 3 matched-pair dataset generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docdet.config import SPECIMEN_LABEL, load_dataset_config, resolve_project_path
from docdet.data.degrade import apply_degradation, sample_degradation
from docdet.data.identities import load_identities
from docdet.data.render import render_document
from docdet.data.tamper import (
    DIFFICULTIES,
    TAMPER_CATEGORIES,
    TARGET_FIELDS,
    null_edit,
    tamper_specimen,
)
from docdet.data.templates import CANVAS_SIZE
from docdet.seeding import derive_seed

CAPTURE_SETTINGS = {
    "canvas_size": [CANVAS_SIZE[0], CANVAS_SIZE[1]],
    "color_mode": "RGB",
    "specimen_label": SPECIMEN_LABEL,
    "geometric_transform": "none",
    "crop": "none",
}


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
        object.__setattr__(self, "manifest_path", resolve_project_path(self.manifest_path))


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
    categories = TAMPER_CATEGORIES
    difficulties = DIFFICULTIES
    templates = tuple(sorted({row["template_id"] for row in manifest}))
    splits = tuple(sorted({row["split"] for row in manifest}))
    return {
        "image_count": len(manifest),
        "label_counts": {label: sum(row["label"] == label for row in manifest) for label in labels},
        "category_counts": {
            category: sum(row["category"] == category for row in manifest)
            for category in categories
        },
        "difficulty_counts": {
            difficulty: sum(row["difficulty"] == difficulty for row in manifest)
            for difficulty in difficulties
        },
        "template_counts": {
            template: sum(row["template_id"] == template for row in manifest)
            for template in templates
        },
        "split_counts": {
            split: sum(row["split"] == split for row in manifest) for split in splits
        },
    }


def _difficulty_for(identity_index: int, category_index: int) -> str:
    return DIFFICULTIES[(identity_index + category_index) % len(DIFFICULTIES)]


def _manifest_entry(
    config: GenerationConfig,
    record: Mapping[str, Any],
    pair_id: str,
    image_id: str,
    label: str,
    category: str,
    difficulty: str,
    target_field: str,
    result: Any,
    degradation: Any,
    degradation_seed: int,
    pair_seed: int,
    tamper_seed: int,
) -> dict[str, Any]:
    split = config.splits[record["identity_id"]]
    entry: dict[str, Any] = {
        "schema_version": 1,
        "image_id": image_id,
        "pair_id": pair_id,
        "base_render_id": f"{record['identity_id']}__base",
        "group_id": record["group_id"],
        "identity_id": record["identity_id"],
        "template_id": record["template_id"],
        "template_family": record["template_family"],
        "split": split,
        "label": label,
        "category": category,
        "difficulty": difficulty,
        "target_field": target_field,
        "image_path": str(Path("images") / split / f"{image_id}.png"),
        "mask_path": str(Path("masks") / split / f"{image_id}__mask.png"),
        "pair_seed": pair_seed,
        "tamper_seed": tamper_seed,
        "degradation_seed": degradation_seed,
        "degradation_params": degradation.to_dict(),
        "capture_settings": dict(CAPTURE_SETTINGS),
        "specimen_label": record["specimen_label"],
        "identifier_style": record["identifier_style"],
    }
    entry.update(result.metadata)
    entry["image_id"] = image_id
    entry["pair_id"] = pair_id
    entry["label"] = label
    entry["category"] = category
    entry["difficulty"] = difficulty
    entry["target_field"] = target_field
    entry["operation"] = result.metadata["operation"]
    return entry


def generate_dataset(config: GenerationConfig) -> GenerationResult:
    output_dir = config.output_dir
    manifest_path = config.manifest_path
    for split in ("train", "validation", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split).mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    used_identifiers: set[str] = {
        record["identifier"] for record in config.identities
    }
    manifest: list[dict[str, Any]] = []

    for identity_index, record in enumerate(config.identities):
        identity_id = record["identity_id"]
        if identity_id not in config.splits:
            raise ValueError(f"Missing split assignment for identity: {identity_id}")
        specimen = render_document(record)

        for category_index, category in enumerate(TAMPER_CATEGORIES):
            difficulty = _difficulty_for(identity_index, category_index)
            pair_id = f"{identity_id}__{category}__{difficulty}"
            pair_seed = derive_seed(
                config.seed, "pair", identity_id, category, difficulty
            )
            tamper_seed = derive_seed(
                config.seed, "tamper", identity_id, category, difficulty
            )
            degradation_seed = derive_seed(
                config.seed, "degradation", identity_id, pair_id, difficulty
            )
            degradation = sample_degradation(
                degradation_seed, identity_id, pair_id, difficulty
            )

            genuine = null_edit(
                specimen, record, category, difficulty, pair_seed
            )
            tampered = tamper_specimen(
                specimen,
                record,
                category,
                difficulty,
                tamper_seed,
                used_identifiers=used_identifiers,
            )
            if category == "DATE_NUMBER":
                used_identifiers.add(tampered.metadata["identifier"])

            for label, result in (("GENUINE", genuine), ("TAMPERED", tampered)):
                image_id = f"{pair_id}__{label}"
                image_path = output_dir / "images" / config.splits[identity_id] / f"{image_id}.png"
                mask_path = output_dir / "masks" / config.splits[identity_id] / f"{image_id}__mask.png"
                degraded = apply_degradation(result.image, degradation)
                degraded.save(image_path, format="PNG")
                result.mask.save(mask_path, format="PNG")
                manifest.append(
                    _manifest_entry(
                        config,
                        record,
                        pair_id,
                        image_id,
                        label,
                        category,
                        difficulty,
                        TARGET_FIELDS[category],
                        result,
                        degradation,
                        degradation_seed,
                        pair_seed,
                        tamper_seed,
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
        description="Generate Phase 3 synthetic matched-pair dataset"
    )
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument(
        "--identities", default="data/identities/identities.json"
    )
    parser.add_argument("--splits", default="data/splits/splits_A.csv")
    parser.add_argument("--output-dir", default="data/generated")
    parser.add_argument("--manifest", default="data/manifests/dataset.jsonl")
    parser.add_argument("--per-template", type=int, default=None)
    args = parser.parse_args(argv)

    if args.per_template is not None and args.per_template <= 0:
        raise ValueError("--per-template must be positive")

    records = load_identities(args.identities)
    if args.per_template is not None:
        counts: Counter[str] = Counter()
        selected: list[dict[str, Any]] = []
        for record in records:
            template_id = record["template_id"]
            if counts[template_id] < args.per_template:
                selected.append(record)
                counts[template_id] += 1
        records = selected

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
        f"Wrote {stats['image_count']} images and "
        f"{stats['image_count']} masks to {result.output_dir}"
    )
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAPTURE_SETTINGS",
    "GenerationConfig",
    "GenerationResult",
    "dataset_statistics",
    "generate_dataset",
    "load_split_map",
]
