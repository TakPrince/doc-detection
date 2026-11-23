from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.data.identities import load_identities  # noqa: E402
from docdet.data.render import render_document  # noqa: E402
from docdet.data.tamper import (  # noqa: E402
    DIFFICULTIES,
    TAMPER_CATEGORIES,
    TARGET_FIELDS,
    identifier_is_invalid,
    null_edit,
    tamper_specimen,
)
from docdet.seeding import derive_seed  # noqa: E402


@pytest.fixture(scope="module")
def identity_records() -> list[dict]:
    records = load_identities(PROJECT_ROOT / "data" / "identities" / "identities.json")
    by_template: dict[str, dict] = {}
    for record in records:
        by_template.setdefault(record["template_id"], record)
    return [by_template[template_id] for template_id in sorted(by_template)]


@pytest.fixture(scope="module")
def rendered_records(identity_records):
    return [(record, render_document(record)) for record in identity_records]


def _seed(record: dict, category: str, difficulty: str) -> int:
    return derive_seed(20261124, "test", record["identity_id"], category, difficulty)


def _used_identifiers(records: list[dict]) -> set[str]:
    return {record["identifier"] for record in records}


@pytest.mark.parametrize("category", TAMPER_CATEGORIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_null_edit_preserves_image_and_creates_empty_mask(
    rendered_records, category, difficulty
):
    record, specimen = rendered_records[0]
    result = null_edit(specimen, record, category, difficulty, _seed(record, category, difficulty))

    assert result.image.tobytes() == specimen.image.tobytes()
    assert result.mask.mode == "L"
    assert result.mask.getbbox() is None
    assert result.metadata["operation"] == "NULL_EDIT"
    assert result.metadata["category"] == category
    assert result.metadata["difficulty"] == difficulty
    assert result.metadata["target_field"] == TARGET_FIELDS[category]
    assert result.metadata["changed"] is False


@pytest.mark.parametrize("category", TAMPER_CATEGORIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_tampered_mask_marks_changed_region(rendered_records, category, difficulty):
    record, specimen = rendered_records[0]
    used = _used_identifiers([row for row, _ in rendered_records])
    result = tamper_specimen(
        specimen,
        record,
        category,
        difficulty,
        _seed(record, category, difficulty),
        used_identifiers=used,
    )

    assert result.image.size == specimen.image.size
    assert result.mask.mode == "L"
    assert result.mask.getbbox() is not None
    assert result.metadata["operation"] == "TAMPER"
    assert result.metadata["changed"] is True
    assert result.metadata["category"] == category
    assert result.metadata["difficulty"] == difficulty
    assert result.metadata["target_field"] == TARGET_FIELDS[category]


@pytest.mark.parametrize("category", TAMPER_CATEGORIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_pre_degradation_changes_are_confined_to_mask(rendered_records, category, difficulty):
    record, specimen = rendered_records[0]
    used = _used_identifiers([row for row, _ in rendered_records])
    result = tamper_specimen(
        specimen,
        record,
        category,
        difficulty,
        _seed(record, category, difficulty),
        used_identifiers=used,
    )
    original = specimen.image.load()
    changed = result.image.load()
    mask = result.mask.load()

    for y in range(specimen.image.height):
        for x in range(specimen.image.width):
            if mask[x, y] == 0:
                assert original[x, y] == changed[x, y], (category, difficulty, x, y)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_identifier_tampering_keeps_identifier_invalid_and_unique(
    identity_records, rendered_records, difficulty
):
    records = [row for row, _ in rendered_records]
    used = _used_identifiers(records)
    record, specimen = rendered_records[0]
    original_identifier = record["identifier"]

    result = tamper_specimen(
        specimen,
        record,
        "DATE_NUMBER",
        difficulty,
        _seed(record, "DATE_NUMBER", difficulty),
        used_identifiers=used,
    )
    replacement = result.metadata["identifier"]

    assert replacement != original_identifier
    assert replacement not in used
    assert identifier_is_invalid(replacement, record["template_family"])
    assert result.metadata["original_identifier"] == original_identifier


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_photo_tampering_replaces_avatar_with_different_avatar(rendered_records, difficulty):
    record, specimen = rendered_records[0]
    result = tamper_specimen(
        specimen,
        record,
        "PHOTO",
        difficulty,
        _seed(record, "PHOTO", difficulty),
    )
    left, top, width, height = specimen.avatar_box
    original_crop = specimen.image.crop((left, top, left + width, top + height))
    changed_crop = result.image.crop((left, top, left + width, top + height))

    assert original_crop.tobytes() != changed_crop.tobytes()
    assert result.metadata["avatar_seed"] != record["avatar_seed"]
    assert result.mask.getbbox() == specimen.avatar_box[:2] + (
        specimen.avatar_box[0] + specimen.avatar_box[2],
        specimen.avatar_box[1] + specimen.avatar_box[3],
    )


def test_text_tampering_keeps_fictional_name_format(rendered_records):
    record, specimen = rendered_records[0]
    result = tamper_specimen(
        specimen,
        record,
        "TEXT",
        "EASY",
        _seed(record, "TEXT", "EASY"),
    )

    assert result.metadata["field_value"].startswith("Sample Name ")
    assert result.metadata["field_value"] != record["fields"]["full_name"]


def test_tamper_does_not_mutate_identity_or_render(rendered_records):
    record, specimen = rendered_records[0]
    original_record = copy.deepcopy(record)
    original_image = specimen.image.tobytes()

    tamper_specimen(
        specimen,
        record,
        "TEXT",
        "MEDIUM",
        _seed(record, "TEXT", "MEDIUM"),
    )

    assert record == original_record
    assert specimen.image.tobytes() == original_image


@pytest.mark.parametrize(
    ("category", "difficulty"),
    [("AUDIO", "EASY"), ("TEXT", "HARD"), ("", "EASY")],
)
def test_tamper_rejects_unknown_category_or_difficulty(rendered_records, category, difficulty):
    record, specimen = rendered_records[0]
    with pytest.raises(ValueError):
        tamper_specimen(specimen, record, category, difficulty, 1)
