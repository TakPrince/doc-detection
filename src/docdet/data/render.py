"""Pillow rendering for synthetic specimens with stable overlays and metadata."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from docdet.config import DEFAULT_TEMPLATES, SPECIMEN_LABEL
from docdet.data.avatars import render_avatar
from docdet.data.identities import load_identities
from docdet.data.templates import (
    CANVAS_SIZE,
    FIELD_KEYS,
    BBox,
    FieldSpec,
    TemplateSpec,
    get_template,
)

OVERLAY_BBOX: BBox = (0, 563, 1000, 62)
OVERLAY_BACKGROUND = (25, 25, 28)
OVERLAY_FOREGROUND = (248, 248, 246)
_FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
_FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
_LABEL_FONT_SIZE = 14
_TITLE_FONT_SIZE = 30
_SUBTITLE_FONT_SIZE = 16
_OVERLAY_FONT_SIZE = 24


@dataclass(frozen=True)
class RenderedSpecimen:
    image: Image.Image
    template_id: str
    field_boxes: dict[str, BBox]
    avatar_box: BBox
    overlay_box: BBox
    specimen_label: str
    metadata: dict[str, Any]


@lru_cache(maxsize=None)
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _FONT_BOLD if bold else _FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def load_font(size: int, bold: bool = False):
    return _load_font(size, bold)


def _coerce_identity(identity: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(identity, Mapping):
        record = dict(identity)
    elif hasattr(identity, "to_dict"):
        record = dict(identity.to_dict())
    else:
        raise TypeError("identity must be a mapping or an Identity instance")

    fields = record.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("identity fields must be a mapping")
    record["fields"] = dict(fields)
    return record


def _validate_avatar_seed(seed: Any) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("avatar_seed must be an integer")
    if seed < 0:
        raise ValueError("avatar_seed must be non-negative")
    return seed


def _field_value(record: Mapping[str, Any], key: str) -> str:
    if key == "identifier":
        value = record.get("identifier")
    else:
        value = record["fields"].get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"identity is missing required field: {key}")
    return value


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    lines = _wrap_text(draw, text, font, max_width)
    if len(lines) <= max_lines:
        return lines

    lines = lines[:max_lines]
    last = lines[-1]
    while last and draw.textlength(f"{last}…", font=font) > max_width:
        last = last[:-1]
    lines[-1] = f"{last.rstrip()}…"
    return lines


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    field: FieldSpec,
    color: tuple[int, int, int],
) -> None:
    left, top, width, height = field.bbox
    font = _load_font(field.font_size, field.bold)
    lines = _fit_lines(draw, text, font, width, field.max_lines)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 2
    available_lines = max(1, height // line_height)
    lines = lines[:available_lines]

    y = top
    for line in lines:
        draw.text((left, y), line, font=font, fill=color, anchor="lt")
        y += line_height


def _draw_template_background(
    draw: ImageDraw.ImageDraw,
    template: TemplateSpec,
) -> None:
    if template.side_rail_bbox is not None:
        _fill_bbox(draw, template.side_rail_bbox, template.accent)
    if template.header_bbox is not None:
        _fill_bbox(draw, template.header_bbox, template.accent)

    left, top, width, height = template.panel_bbox
    draw.rounded_rectangle(
        (left, top, left + width - 1, top + height - 1),
        radius=22,
        fill=template.panel,
        outline=template.accent,
        width=2,
    )

    if template.rule_bbox is not None:
        _fill_bbox(draw, template.rule_bbox, template.accent)

    draw.text(
        template.title_origin,
        template.title,
        font=_load_font(_TITLE_FONT_SIZE, True),
        fill=template.title_color,
        anchor="lt",
    )
    draw.text(
        template.subtitle_origin,
        template.subtitle,
        font=_load_font(_SUBTITLE_FONT_SIZE),
        fill=template.subtitle_color,
        anchor="lt",
    )


def _fill_bbox(
    draw: ImageDraw.ImageDraw,
    box: BBox,
    color: tuple[int, int, int],
) -> None:
    left, top, width, height = box
    draw.rectangle((left, top, left + width - 1, top + height - 1), fill=color)


def _draw_avatar(
    image: Image.Image,
    template: TemplateSpec,
    avatar_seed: int,
) -> None:
    left, top, width, height = template.avatar_bbox
    avatar = render_avatar(avatar_seed, (width, height))
    radius = max(8, min(width, height) // 10)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=radius,
        fill=255,
    )
    image.paste(avatar, (left, top), mask)
    ImageDraw.Draw(image).rounded_rectangle(
        (left, top, left + width - 1, top + height - 1),
        radius=radius,
        outline=template.accent,
        width=3,
    )


def _draw_fields(
    draw: ImageDraw.ImageDraw,
    template: TemplateSpec,
    record: Mapping[str, Any],
) -> None:
    label_font = _load_font(_LABEL_FONT_SIZE, True)
    for field in template.fields:
        left, top, _, _ = field.bbox
        draw.text(
            (left, top - 24),
            field.label,
            font=label_font,
            fill=template.muted,
            anchor="lt",
        )
        _draw_text_block(draw, _field_value(record, field.key), field, template.text)


def _draw_overlay(draw: ImageDraw.ImageDraw) -> None:
    left, top, width, height = OVERLAY_BBOX
    draw.rectangle(
        (left, top, left + width - 1, top + height - 1),
        fill=OVERLAY_BACKGROUND,
    )
    draw.text(
        (left + width / 2, top + height / 2),
        SPECIMEN_LABEL,
        font=_load_font(_OVERLAY_FONT_SIZE, True),
        fill=OVERLAY_FOREGROUND,
        anchor="mm",
    )


def render_document(identity: Mapping[str, Any] | Any) -> RenderedSpecimen:
    record = _coerce_identity(identity)
    template_id = record.get("template_id")
    if not isinstance(template_id, str):
        raise ValueError("identity template_id must be a string")
    template = get_template(template_id)

    specimen_label = record.get("specimen_label", SPECIMEN_LABEL)
    if specimen_label != SPECIMEN_LABEL:
        raise ValueError("identity specimen label is invalid")

    avatar_seed = _validate_avatar_seed(record.get("avatar_seed"))
    for key in FIELD_KEYS:
        _field_value(record, key)

    image = Image.new("RGB", CANVAS_SIZE, template.background)
    draw = ImageDraw.Draw(image)
    _draw_template_background(draw, template)
    _draw_avatar(image, template, avatar_seed)
    _draw_fields(draw, template, record)
    _draw_overlay(draw)

    field_boxes = {field.key: field.bbox for field in template.fields}
    metadata = {
        "schema_version": 1,
        "identity_id": record.get("identity_id"),
        "group_id": record.get("group_id"),
        "template_id": template.template_id,
        "template_family": template.family,
        "size": [CANVAS_SIZE[0], CANVAS_SIZE[1]],
        "specimen_label": SPECIMEN_LABEL,
        "avatar_seed": avatar_seed,
        "field_keys": list(FIELD_KEYS),
    }

    return RenderedSpecimen(
        image=image,
        template_id=template.template_id,
        field_boxes=field_boxes,
        avatar_box=template.avatar_bbox,
        overlay_box=OVERLAY_BBOX,
        specimen_label=SPECIMEN_LABEL,
        metadata=metadata,
    )


def save_specimen(result: RenderedSpecimen, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(path, format="PNG")
    return path


def render_gallery(
    identities: Sequence[Mapping[str, Any] | Any],
    output_dir: str | Path,
    per_template: int = 1,
) -> list[Path]:
    if not isinstance(per_template, int) or isinstance(per_template, bool):
        raise ValueError("per_template must be an integer")
    if per_template <= 0:
        raise ValueError("per_template must be positive")

    records = [_coerce_identity(identity) for identity in identities]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for template_id in DEFAULT_TEMPLATES:
        selected = [
            record for record in records if record.get("template_id") == template_id
        ][:per_template]
        if not selected:
            raise ValueError(f"gallery is missing a specimen for template {template_id}")
        for record in selected:
            identity_id = record.get("identity_id")
            if not isinstance(identity_id, str) or not identity_id:
                raise ValueError("gallery identity_id must be a non-empty string")
            result = render_document(record)
            paths.append(save_specimen(result, destination / f"{identity_id}.png"))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a small Phase 2 synthetic specimen gallery"
    )
    parser.add_argument(
        "--identities",
        default="data/identities/identities.json",
    )
    parser.add_argument(
        "--output",
        default="docs/figures/phase2_gallery",
    )
    parser.add_argument("--per-template", type=int, default=1)
    args = parser.parse_args(argv)

    records = load_identities(args.identities)
    paths = render_gallery(records, args.output, per_template=args.per_template)
    print(f"Wrote {len(paths)} gallery specimens to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANVAS_SIZE",
    "OVERLAY_BBOX",
    "RenderedSpecimen",
    "load_font",
    "render_document",
    "render_gallery",
    "save_specimen",
]
