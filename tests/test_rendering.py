from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.config import DEFAULT_TEMPLATES, SPECIMEN_LABEL  # noqa: E402
from docdet.data.avatars import render_avatar  # noqa: E402
from docdet.data.identities import load_identities  # noqa: E402
from docdet.data.render import (  # noqa: E402
    CANVAS_SIZE,
    OVERLAY_BBOX,
    render_document,
    render_gallery,
)
from docdet.data.templates import FIELD_KEYS, TEMPLATES, get_template  # noqa: E402


def _inside_canvas(box: tuple[int, int, int, int]) -> bool:
    left, top, width, height = box
    return (
        left >= 0
        and top >= 0
        and width > 0
        and height > 0
        and left + width <= CANVAS_SIZE[0]
        and top + height <= CANVAS_SIZE[1]
    )


@pytest.fixture(scope="module")
def sample_identities() -> list[dict]:
    records = load_identities(PROJECT_ROOT / "data" / "identities" / "identities.json")
    by_template: dict[str, dict] = {}
    for record in records:
        by_template.setdefault(record["template_id"], record)
    return [by_template[template_id] for template_id in DEFAULT_TEMPLATES]


def test_avatar_rendering_is_deterministic_and_size_sensitive():
    first = render_avatar(12345, (120, 160))
    second = render_avatar(12345, (120, 160))
    different = render_avatar(54321, (120, 160))

    assert first.size == (120, 160)
    assert first.mode == "RGB"
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != different.tobytes()


def test_avatar_rejects_invalid_seed_and_size():
    with pytest.raises(ValueError):
        render_avatar(-1)
    with pytest.raises(ValueError):
        render_avatar(1, (0, 160))
    with pytest.raises(ValueError):
        render_avatar(1, (120, -1))


def test_all_four_templates_have_required_fields_and_valid_boxes():
    assert tuple(TEMPLATES) == DEFAULT_TEMPLATES

    for template_id, template in TEMPLATES.items():
        assert template.template_id == template_id
        assert {field.key for field in template.fields} == set(FIELD_KEYS)
        assert _inside_canvas(template.avatar_bbox)
        assert template.avatar_bbox[1] + template.avatar_bbox[3] < OVERLAY_BBOX[1]
        for field in template.fields:
            assert field.label
            assert field.font_size > 0
            assert _inside_canvas(field.bbox)
            assert field.bbox[1] + field.bbox[3] < OVERLAY_BBOX[1]


def test_templates_have_distinct_field_layouts():
    signatures = {
        tuple((field.key, field.bbox) for field in template.fields)
        for template in TEMPLATES.values()
    }
    assert len(signatures) == len(DEFAULT_TEMPLATES)


def test_get_template_rejects_unknown_template():
    with pytest.raises(ValueError):
        get_template("AAD-9")


def test_render_document_outputs_required_metadata(sample_identities):
    for record in sample_identities:
        result = render_document(record)

        assert result.image.size == CANVAS_SIZE
        assert result.image.mode == "RGB"
        assert result.template_id == record["template_id"]
        assert result.specimen_label == SPECIMEN_LABEL
        assert result.overlay_box == OVERLAY_BBOX
        assert result.avatar_box == get_template(record["template_id"]).avatar_bbox
        assert set(result.field_boxes) == set(FIELD_KEYS)
        assert result.metadata["identity_id"] == record["identity_id"]
        assert result.metadata["template_id"] == record["template_id"]
        assert result.metadata["specimen_label"] == SPECIMEN_LABEL

        for box in result.field_boxes.values():
            assert _inside_canvas(box)
            assert box[1] + box[3] <= OVERLAY_BBOX[1]


def test_render_field_regions_contain_text(sample_identities):
    for record in sample_identities:
        result = render_document(record)
        for left, top, width, height in result.field_boxes.values():
            crop = result.image.crop((left, top, left + width, top + height))
            colors = crop.getcolors(maxcolors=1_000_000)
            assert colors is not None
            assert len(colors) > 4


def test_render_watermark_is_visible_and_stable(sample_identities):
    for record in sample_identities:
        result = render_document(record)
        left, top, width, height = result.overlay_box
        crop = result.image.crop((left, top, left + width, top + height))
        colors = crop.getcolors(maxcolors=1_000_000)

        assert result.metadata["specimen_label"] == SPECIMEN_LABEL
        assert colors is not None
        assert len(colors) > 8


def test_render_is_deterministic_and_does_not_mutate_identity(sample_identities):
    record = sample_identities[0]
    before = copy.deepcopy(record)

    first = render_document(record)
    second = render_document(record)

    assert record == before
    assert first.image.tobytes() == second.image.tobytes()
    assert first.field_boxes == second.field_boxes
    assert first.overlay_box == second.overlay_box


def test_render_rejects_unknown_template_and_invalid_avatar_seed(sample_identities):
    unknown_template = copy.deepcopy(sample_identities[0])
    unknown_template["template_id"] = "AAD-9"
    with pytest.raises(ValueError):
        render_document(unknown_template)

    invalid_seed = copy.deepcopy(sample_identities[0])
    invalid_seed["avatar_seed"] = -1
    with pytest.raises(ValueError):
        render_document(invalid_seed)


def test_gallery_writes_one_specimen_per_template(tmp_path, sample_identities):
    paths = render_gallery(sample_identities, tmp_path)

    assert len(paths) == len(DEFAULT_TEMPLATES)
    assert {path.name for path in paths} == {
        f"{record['identity_id']}.png" for record in sample_identities
    }
    for path in paths:
        with Image.open(path) as image:
            assert image.size == CANVAS_SIZE
            assert image.mode == "RGB"
