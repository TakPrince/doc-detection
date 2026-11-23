"""Deterministic visual tampering with binary masks and provenance."""

from __future__ import annotations

import random
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from docdet.data.avatars import render_avatar
from docdet.data.identities import (
    AADHAAR_PATTERN,
    INVALID_PAN_CATEGORY_LETTERS,
    PAN_PATTERN,
    RECOGNIZED_PAN_CATEGORIES,
    verhoeff_is_valid,
)
from docdet.data.render import RenderedSpecimen, load_font
from docdet.data.templates import BBox, FieldSpec, TemplateSpec, get_template

CATEGORY_TEXT = "TEXT"
CATEGORY_DATE_NUMBER = "DATE_NUMBER"
CATEGORY_PHOTO = "PHOTO"
TAMPER_CATEGORIES = (CATEGORY_TEXT, CATEGORY_DATE_NUMBER, CATEGORY_PHOTO)

DIFFICULTY_EASY = "EASY"
DIFFICULTY_MEDIUM = "MEDIUM"
DIFFICULTIES = (DIFFICULTY_EASY, DIFFICULTY_MEDIUM)

TARGET_FIELDS = {
    CATEGORY_TEXT: "full_name",
    CATEGORY_DATE_NUMBER: "identifier",
    CATEGORY_PHOTO: "avatar",
}
_IDENTIFIER_MIN_DISTANCE = {"EASY": 6, "MEDIUM": 3}
_NAME_MIN_DISTANCE = {"EASY": 3, "MEDIUM": 1}
_NAME_PREFIX = "Sample Name "


@dataclass(frozen=True)
class TamperResult:
    image: Image.Image
    mask: Image.Image
    metadata: dict[str, Any]


def _validate_category(category: str) -> None:
    if category not in TAMPER_CATEGORIES:
        raise ValueError(f"Unknown tamper category: {category}")


def _validate_difficulty(difficulty: str) -> None:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("tamper seed must be an integer")
    if seed < 0:
        raise ValueError("tamper seed must be non-negative")


def identifier_is_invalid(identifier: str, family: str) -> bool:
    if not isinstance(identifier, str):
        return False
    if family == "AAD":
        return bool(AADHAAR_PATTERN.fullmatch(identifier)) and not verhoeff_is_valid(
            identifier
        )
    if family == "PAN":
        return bool(PAN_PATTERN.fullmatch(identifier)) and (
            identifier[3] not in RECOGNIZED_PAN_CATEGORIES
        )
    return False


def _hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        return abs(len(left) - len(right)) + min(len(left), len(right))
    return sum(first != second for first, second in zip(left, right))


def _replacement_name(rng: random.Random, original: str, difficulty: str) -> str:
    if not original.startswith(_NAME_PREFIX):
        raise ValueError("full_name must use the synthetic sample format")
    original_suffix = original[len(_NAME_PREFIX) :]
    threshold = _NAME_MIN_DISTANCE[difficulty]

    for _ in range(10_000):
        suffix = "".join(str(rng.randrange(10)) for _ in range(4))
        if suffix == original_suffix:
            continue
        if _hamming_distance(suffix, original_suffix) < threshold:
            continue
        return f"{_NAME_PREFIX}{suffix}"
    raise RuntimeError("Could not generate a replacement fictional name")


def _replacement_identifier(
    rng: random.Random,
    family: str,
    original: str,
    used_identifiers: set[str],
    difficulty: str,
) -> str:
    threshold = _IDENTIFIER_MIN_DISTANCE[difficulty]

    for _ in range(10_000):
        if family == "AAD":
            candidate = rng.choice(("0", "1")) + "".join(
                str(rng.randrange(10)) for _ in range(11)
            )
        elif family == "PAN":
            candidate = (
                "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
                + rng.choice(INVALID_PAN_CATEGORY_LETTERS)
                + rng.choice(string.ascii_uppercase)
                + "".join(str(rng.randrange(10)) for _ in range(4))
                + rng.choice(string.ascii_uppercase)
            )
        else:
            raise ValueError(f"Unknown template family: {family}")

        if not identifier_is_invalid(candidate, family):
            continue
        if candidate == original or candidate in used_identifiers:
            continue
        if _hamming_distance(candidate, original) < threshold:
            continue
        return candidate
    raise RuntimeError("Could not generate a unique invalid replacement identifier")


def _field_spec(specimen: RenderedSpecimen, key: str) -> FieldSpec:
    template = get_template(specimen.template_id)
    return template.field_map[key]


def _redraw_value(
    image: Image.Image,
    template: TemplateSpec,
    field: FieldSpec,
    value: str,
) -> None:
    left, top, width, height = field.bbox
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left, top, left + width - 1, top + height - 1),
        fill=template.panel,
    )
    draw.text(
        (left, top),
        value,
        font=load_font(field.font_size, field.bold),
        fill=template.text,
        anchor="lt",
    )


def _box_mask(size: tuple[int, int], box: BBox) -> Image.Image:
    mask = Image.new("L", size, 0)
    left, top, width, height = box
    ImageDraw.Draw(mask).rectangle(
        (left, top, left + width - 1, top + height - 1),
        fill=255,
    )
    return mask


def _paste_new_avatar(
    image: Image.Image,
    template: TemplateSpec,
    avatar_seed: int,
    box: BBox,
) -> None:
    left, top, width, height = box
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


def _current_field_value(identity: Mapping[str, Any], key: str) -> str | None:
    if key == "identifier":
        return identity.get("identifier")
    if key == "avatar":
        return None
    fields = identity.get("fields", {})
    if isinstance(fields, Mapping):
        return fields.get(key)
    return None


def _base_metadata(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    difficulty: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "identity_id": identity.get("identity_id"),
        "group_id": identity.get("group_id"),
        "template_id": specimen.template_id,
        "template_family": identity.get("template_family"),
        "category": category,
        "difficulty": difficulty,
        "target_field": TARGET_FIELDS[category],
        "seed": seed,
        "base_render_seed": identity.get("avatar_seed"),
        "avatar_seed": identity.get("avatar_seed"),
        "identifier": identity.get("identifier"),
        "original_identifier": identity.get("identifier"),
        "field_value": _current_field_value(identity, TARGET_FIELDS[category]),
    }


def null_edit(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    difficulty: str,
    seed: int,
) -> TamperResult:
    _validate_category(category)
    _validate_difficulty(difficulty)
    _validate_seed(seed)
    metadata = _base_metadata(specimen, identity, category, difficulty, seed)
    metadata.update(
        operation="NULL_EDIT",
        changed=False,
        mask_bbox=None,
        specimen_label=specimen.specimen_label,
    )
    return TamperResult(
        image=specimen.image.copy(),
        mask=Image.new("L", specimen.image.size, 0),
        metadata=metadata,
    )


def tamper_specimen(
    specimen: RenderedSpecimen,
    identity: Mapping[str, Any],
    category: str,
    difficulty: str,
    seed: int,
    used_identifiers: set[str] | None = None,
) -> TamperResult:
    _validate_category(category)
    _validate_difficulty(difficulty)
    _validate_seed(seed)

    template = get_template(specimen.template_id)
    rng = random.Random(seed)
    image = specimen.image.copy()
    metadata = _base_metadata(specimen, identity, category, difficulty, seed)

    if category == CATEGORY_TEXT:
        field = _field_spec(specimen, "full_name")
        original = identity["fields"]["full_name"]
        replacement = _replacement_name(rng, original, difficulty)
        _redraw_value(image, template, field, replacement)
        mask = _box_mask(image.size, field.bbox)
        metadata.update(
            field_value=replacement,
            original_field_value=original,
            identifier=identity["identifier"],
        )
    elif category == CATEGORY_DATE_NUMBER:
        field = _field_spec(specimen, "identifier")
        original = identity["identifier"]
        reserved = set(used_identifiers or ())
        replacement = _replacement_identifier(
            rng,
            identity["template_family"],
            original,
            reserved,
            difficulty,
        )
        _redraw_value(image, template, field, replacement)
        mask = _box_mask(image.size, field.bbox)
        metadata.update(
            identifier=replacement,
            original_identifier=original,
            identifier_style=identity["identifier_style"],
            field_value=replacement,
        )
    else:
        box = template.avatar_bbox
        original_seed = identity["avatar_seed"]
        replacement_seed = rng.randrange(1, 2**31)
        while replacement_seed == original_seed:
            replacement_seed = rng.randrange(1, 2**31)
        _paste_new_avatar(image, template, replacement_seed, box)
        mask = _box_mask(image.size, box)
        metadata.update(
            avatar_seed=replacement_seed,
            original_avatar_seed=original_seed,
            field_value=None,
        )

    metadata.update(
        operation="TAMPER",
        changed=True,
        mask_bbox=list(mask.getbbox() or ()),
        specimen_label=specimen.specimen_label,
    )
    return TamperResult(image=image, mask=mask, metadata=metadata)


__all__ = [
    "CATEGORY_DATE_NUMBER",
    "CATEGORY_PHOTO",
    "CATEGORY_TEXT",
    "DIFFICULTIES",
    "TAMPER_CATEGORIES",
    "TARGET_FIELDS",
    "TamperResult",
    "identifier_is_invalid",
    "null_edit",
    "tamper_specimen",
]
