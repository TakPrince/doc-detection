from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.data.generate_dataset import (  # noqa: E402
    GenerationConfig,
    dataset_statistics,
    generate_dataset,
    load_split_map,
)
from docdet.data.identities import load_identities  # noqa: E402
from docdet.data.tamper import DIFFICULTIES, TAMPER_CATEGORIES, identifier_is_invalid  # noqa: E402


@pytest.fixture(scope="module")
def smoke_identities() -> list[dict]:
    records = load_identities(PROJECT_ROOT / "data" / "identities" / "identities.json")
    by_template: dict[str, dict] = {}
    for record in records:
        by_template.setdefault(record["template_id"], record)
    return [by_template[template_id] for template_id in sorted(by_template)]


@pytest.fixture(scope="module")
def split_map() -> dict[str, str]:
    return load_split_map(PROJECT_ROOT / "data" / "splits" / "splits_A.csv")


def _config(tmp_path: Path, identities: list[dict], split_map: dict[str, str]) -> GenerationConfig:
    return GenerationConfig(
        seed=20261124,
        identities=identities,
        splits=split_map,
        output_dir=tmp_path / "generated",
        manifest_path=tmp_path / "manifest.jsonl",
    )


def test_smoke_generation_produces_expected_counts(tmp_path, smoke_identities, split_map):
    result = generate_dataset(_config(tmp_path, smoke_identities, split_map))
    stats = dataset_statistics(result.manifest)

    assert len(result.manifest) == len(smoke_identities) * 6
    assert stats["label_counts"] == {"GENUINE": 12, "TAMPERED": 12}
    assert stats["category_counts"] == {category: 8 for category in TAMPER_CATEGORIES}
    assert sum(stats["difficulty_counts"].values()) == 24
    assert set(stats["difficulty_counts"]) == set(DIFFICULTIES)
    assert set(stats["template_counts"]) == {record["template_id"] for record in smoke_identities}


def test_manifest_has_required_fields_and_existing_assets(tmp_path, smoke_identities, split_map):
    config = _config(tmp_path, smoke_identities, split_map)
    result = generate_dataset(config)
    required = {
        "image_id",
        "pair_id",
        "group_id",
        "identity_id",
        "template_id",
        "split",
        "label",
        "category",
        "difficulty",
        "mask_path",
        "image_path",
        "target_field",
        "pair_seed",
        "tamper_seed",
        "degradation_seed",
        "degradation_params",
        "capture_settings",
    }

    assert result.manifest_path.exists()
    assert result.manifest_path.read_text(encoding="utf-8").count("\n") == len(result.manifest)

    for entry in result.manifest:
        assert required <= set(entry)
        assert (config.output_dir / entry["image_path"]).exists()
        assert (config.output_dir / entry["mask_path"]).exists()
        assert entry["specimen_label"] == "SPECIMEN – NOT A VALID DOCUMENT"
        assert identifier_is_invalid(entry["identifier"], entry["template_family"])


def test_pairs_share_identity_render_target_and_degradation(tmp_path, smoke_identities, split_map):
    result = generate_dataset(_config(tmp_path, smoke_identities, split_map))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in result.manifest:
        grouped[entry["pair_id"]].append(entry)

    assert len(grouped) == len(smoke_identities) * 3
    for pair_id, members in grouped.items():
        assert pair_id
        assert len(members) == 2
        genuine, tampered = sorted(members, key=lambda row: row["label"])
        assert genuine["label"] == "GENUINE"
        assert tampered["label"] == "TAMPERED"
        for field in (
            "identity_id",
            "group_id",
            "template_id",
            "split",
            "category",
            "difficulty",
            "target_field",
            "base_render_seed",
            "pair_seed",
            "tamper_seed",
            "degradation_seed",
            "degradation_params",
            "capture_settings",
        ):
            assert genuine[field] == tampered[field], (pair_id, field)
        if genuine["category"] == "PHOTO":
            assert genuine["avatar_seed"] == tampered["original_avatar_seed"]
        else:
            assert genuine["avatar_seed"] == tampered["avatar_seed"]
        assert genuine["operation"] == "NULL_EDIT"
        assert tampered["operation"] == "TAMPER"


def test_genuine_masks_are_empty_and_tampered_masks_are_not(tmp_path, smoke_identities, split_map):
    config = _config(tmp_path, smoke_identities, split_map)
    result = generate_dataset(config)

    for entry in result.manifest:
        with Image.open(config.output_dir / entry["mask_path"]) as mask:
            assert mask.mode == "L"
            if entry["label"] == "GENUINE":
                assert mask.getbbox() is None
            else:
                assert mask.getbbox() is not None


def test_split_isolation_and_manifest_split_matches_source(tmp_path, smoke_identities, split_map):
    result = generate_dataset(_config(tmp_path, smoke_identities, split_map))
    identity_splits: dict[str, set[str]] = defaultdict(set)

    for entry in result.manifest:
        identity_splits[entry["identity_id"]].add(entry["split"])
        assert entry["split"] == split_map[entry["identity_id"]]

    assert all(len(splits) == 1 for splits in identity_splits.values())


def test_generation_is_deterministic(tmp_path, smoke_identities, split_map):
    first = generate_dataset(_config(tmp_path / "a", smoke_identities, split_map))
    second = generate_dataset(_config(tmp_path / "b", smoke_identities, split_map))

    assert first.manifest == second.manifest
    assert {entry["image_id"]: entry["image_path"] for entry in first.manifest} == {
        entry["image_id"]: entry["image_path"] for entry in second.manifest
    }
    for left, right in zip(first.manifest, second.manifest):
        assert (first.output_dir / left["image_path"]).read_bytes() == (
            second.output_dir / right["image_path"]
        ).read_bytes()
        assert (first.output_dir / left["mask_path"]).read_bytes() == (
            second.output_dir / right["mask_path"]
        ).read_bytes()


def test_manifest_file_matches_returned_manifest(tmp_path, smoke_identities, split_map):
    result = generate_dataset(_config(tmp_path, smoke_identities, split_map))
    rows = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows == result.manifest
