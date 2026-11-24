from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.data.identities import load_identities  # noqa: E402
from docdet.data.render import render_document  # noqa: E402
from docdet.data.tamper import (  # noqa: E402
    TAMPER_CATEGORIES,
    TARGET_FIELDS,
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


def _seed(record: dict, category: str) -> int:
    return derive_seed(20261124, "test", record["identity_id"], category)


def _target_box(specimen, category: str):
    if category == "PHOTO":
        return specimen.avatar_box
    left, top, width, height = specimen.field_boxes["full_name"]
    return left, top, width, height


@pytest.mark.parametrize("category", TAMPER_CATEGORIES)
def test_null_edit_preserves_image(rendered_records, category):
    record, specimen = rendered_records[0]
    result = null_edit(specimen, record, category, _seed(record, category))

    assert result.image.tobytes() == specimen.image.tobytes()
    assert result.metadata["operation"] == "NULL_EDIT"
    assert result.metadata["category"] == category
    assert result.metadata["target_field"] == TARGET_FIELDS[category]
    assert result.metadata["changed"] is False


@pytest.mark.parametrize("category", TAMPER_CATEGORIES)
def test_tampered_changes_are_confined_to_target_region(rendered_records, category):
    record, specimen = rendered_records[0]
    result = tamper_specimen(specimen, record, category, _seed(record, category))
    original = specimen.image.load()
    changed = result.image.load()
    left, top, width, height = _target_box(specimen, category)

    for y in range(specimen.image.height):
        for x in range(specimen.image.width):
            if left <= x < left + width and top <= y < top + height:
                continue
            assert original[x, y] == changed[x, y], (category, x, y)

    assert result.metadata["operation"] == "TAMPER"
    assert result.metadata["changed"] is True


def test_text_tampering_keeps_fictional_name_format(rendered_records):
    record, specimen = rendered_records[0]
    result = tamper_specimen(specimen, record, "TEXT", _seed(record, "TEXT"))

    assert result.metadata["field_value"].startswith("Sample Name ")
    assert result.metadata["field_value"] != record["fields"]["full_name"]


def test_photo_tampering_replaces_avatar(rendered_records):
    record, specimen = rendered_records[0]
    result = tamper_specimen(specimen, record, "PHOTO", _seed(record, "PHOTO"))
    left, top, width, height = specimen.avatar_box
    original_crop = specimen.image.crop((left, top, left + width, top + height))
    changed_crop = result.image.crop((left, top, left + width, top + height))

    assert original_crop.tobytes() != changed_crop.tobytes()
    assert result.metadata["avatar_seed"] != record["avatar_seed"]


def test_tamper_does_not_mutate_identity_or_render(rendered_records):
    record, specimen = rendered_records[0]
    original_record = copy.deepcopy(record)
    original_image = specimen.image.tobytes()

    tamper_specimen(specimen, record, "TEXT", _seed(record, "TEXT"))

    assert record == original_record
    assert specimen.image.tobytes() == original_image


@pytest.mark.parametrize("category", ["AUDIO", "DATE_NUMBER", ""])
def test_tamper_rejects_unknown_category(rendered_records, category):
    record, specimen = rendered_records[0]
    with pytest.raises(ValueError):
        tamper_specimen(specimen, record, category, 1)
