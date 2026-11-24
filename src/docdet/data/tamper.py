"""Deterministic text and photo tampering for synthetic specimens."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance

from docdet.data.avatars import render_avatar
from docdet.data.render import RenderedSpecimen, load_font
from docdet.data.templates import BBox, FieldSpec, TemplateSpec, get_template

CATEGORY_TEXT = "TEXT"
CATEGORY_PHOTO = "PHOTO"
TAMPER_CATEGORIES = (CATEGORY_TEXT, CATEGORY_PHOTO)
TARGET_FIELDS = {
    CATEGORY_TEXT: "full_name",
    CATEGORY_PHOTO: "avatar",
}
_NAME_PREFIX = "Sample Name "


@dataclass(frozen=True)
class TamperResult:
    image: Image.Image
    metadata: dict[str, Any]


def _validate_category(category: str) -> None:
    if category not in TAMPER_CATEGORIES:
        raise ValueError(f"Unknown tamper category: {category}")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("tamper seed must be an integer")
    if seed < 0:
        raise ValueError("tamper seed must be non-negative")


def _replacement_name(rng: random.Random, original: str) -> str:
    if not original.startswith(_NAME_PREFIX):
        raise ValueError("full_name must use the synthetic sample format")
    original_suffix = original[len(_NAME_PREFIX) :]

    for _ in range(10_000):
        suffix = "".join(str(rng.randrange(10)) for _ in range(4))
        if suffix != original_suffix:
            return f"{_NAME_PREFIX}{suffix}"
    raise RuntimeError("Could not generate a replacement fictional name")


def _redraw_value(
    image: Image.Image,
    template: TemplateSpec,
    field: FieldSpec,
    value: str,
    offset_x: int = 0,
    offset_y: int = 0,
    font_size_delta: int = 0,
    color_delta: tuple[int, int, int] = (0, 0, 0),
) -> None:
    left, top, width, height = field.bbox
    font_size = max(10, field.font_size + font_size_delta)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left, top, left + width - 1, top + height - 1),
        fill=template.panel,
    )
    color = tuple(
        min(255, max(0, channel + delta))
        for channel, delta in zip(template.text, color_delta)
    )
    draw.text(
        (left + offset_x, top + offset_y),
        value,
        font=load_font(font_size, field.bold),
        fill=color,
        anchor="lt",
    )


def _paste_new_avatar(
    image: Image.Image,
    template: TemplateSpec,
    avatar_seed: int,
    box: BBox,
    scale: float = 0.85,
    tone_factor: float = 1.0,
    border_width: int = 3,
    paste_offset: tuple[int, int] = (0, 0),
) -> None:
    left, top, width, height = box
    avatar_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    avatar = render_avatar(avatar_seed, avatar_size)
    if tone_factor != 1.0:
        avatar = ImageEnhance.Brightness(avatar).enhance(tone_factor)
    radius = max(8, min(avatar_size) // 10)
    mask = Image.new("L", avatar_size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, avatar_size[0] - 1, avatar_size[1] - 1),
        radius=radius,
        fill=255,
    )
    paste_x = left + (width - avatar_size[0]) // 2 + paste_offset[0]
    paste_y = top + (height - avatar_size[1]) // 2 + paste_offset[1]
    image.paste(avatar, (paste_x, paste_y), mask)
    ImageDraw.Draw(image).rounded_rectangle(
        (left, top, left + width - 1, top + height - 1),
        radius=max(8, min(width, height) // 10),
        outline=template.accent,
        width=border_width,
    )


def _base_metadata(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "identity_id": identity.get("identity_id"),
        "group_id": identity.get("group_id"),
        "template_id": specimen.template_id,
        "template_family": identity.get("template_family"),
        "category": category,
        "target_field": TARGET_FIELDS[category],
        "seed": seed,
        "identifier": identity.get("identifier"),
    }


def null_edit(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    seed: int,
) -> TamperResult:
    _validate_category(category)
    _validate_seed(seed)
    metadata = _base_metadata(specimen, identity, category, seed)
    metadata.update(
        operation="NULL_EDIT",
        changed=False,
        specimen_label=specimen.specimen_label,
    )
    return TamperResult(image=specimen.image.copy(), metadata=metadata)


def tamper_specimen(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    seed: int,
) -> TamperResult:
    _validate_category(category)
    _validate_seed(seed)

    template = get_template(specimen.template_id)
    rng = random.Random(seed)
    image = specimen.image.copy()
    metadata = _base_metadata(specimen, identity, category, seed)

    if category == CATEGORY_TEXT:
        field = template.field_map["full_name"]
        original = identity["fields"]["full_name"]
        replacement = _replacement_name(rng, original)
        offset_x = rng.randint(0, 8)
        offset_y = rng.randint(-2, 2)
        font_size_delta = rng.randint(-3, 3)
        color_delta = tuple(rng.randint(-55, 55) for _ in range(3))
        _redraw_value(
            image,
            template,
            field,
            replacement,
            offset_x=offset_x,
            offset_y=offset_y,
            font_size_delta=font_size_delta,
            color_delta=color_delta,
        )
        metadata.update(
            original_field_value=original,
            field_value=replacement,
            text_offset_x=offset_x,
            text_offset_y=offset_y,
            text_font_size_delta=font_size_delta,
            text_color_delta=color_delta,
        )
    else:
        box = template.avatar_bbox
        original_seed = identity["avatar_seed"]
        replacement_seed = rng.randrange(1, 2**31)
        while replacement_seed == original_seed:
            replacement_seed = rng.randrange(1, 2**31)
        scale = rng.uniform(0.65, 0.90)
        tone_factor = rng.uniform(0.70, 1.30)
        border_width = rng.randint(1, 5)
        paste_offset = (rng.randint(-4, 4), rng.randint(-4, 4))
        _paste_new_avatar(
            image,
            template,
            replacement_seed,
            box,
            scale=scale,
            tone_factor=tone_factor,
            border_width=border_width,
            paste_offset=paste_offset,
        )
        metadata.update(
            original_avatar_seed=original_seed,
            avatar_seed=replacement_seed,
            photo_scale=scale,
            photo_tone_factor=tone_factor,
            photo_border_width=border_width,
            photo_paste_offset=paste_offset,
        )

    metadata.update(
        operation="TAMPER",
        changed=True,
        specimen_label=specimen.specimen_label,
    )
    return TamperResult(image=image, metadata=metadata)


__all__ = [
    "CATEGORY_PHOTO",
    "CATEGORY_TEXT",
    "TAMPER_CATEGORIES",
    "TARGET_FIELDS",
    "TamperResult",
    "null_edit",
    "tamper_specimen",
]
