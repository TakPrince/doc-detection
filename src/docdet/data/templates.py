"""Fictional layouts; bounding boxes are (left, top, width, height) pixels."""

from __future__ import annotations

from dataclasses import dataclass

from docdet.config import DEFAULT_TEMPLATES

BBox = tuple[int, int, int, int]
RGB = tuple[int, int, int]
CANVAS_SIZE = (1000, 625)
FIELD_KEYS = ("identifier", "full_name", "date_of_birth", "address")
FIELD_LABELS = {
    "identifier": "IDENTIFIER",
    "full_name": "FULL NAME",
    "date_of_birth": "DATE OF BIRTH",
    "address": "ADDRESS",
}


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    bbox: BBox
    font_size: int
    max_lines: int = 1
    bold: bool = False

    def __post_init__(self) -> None:
        if self.key not in FIELD_KEYS:
            raise ValueError(f"Unknown field key: {self.key}")
        if self.label != FIELD_LABELS[self.key]:
            raise ValueError(f"Field label mismatch for {self.key}")
        if self.font_size <= 0:
            raise ValueError("Field font size must be positive")
        if self.max_lines <= 0:
            raise ValueError("Field max_lines must be positive")
        _validate_bbox(self.bbox)


@dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str
    title: str
    subtitle: str
    background: RGB
    panel: RGB
    accent: RGB
    text: RGB
    muted: RGB
    panel_bbox: BBox
    avatar_bbox: BBox
    fields: tuple[FieldSpec, ...]
    header_bbox: BBox | None = None
    side_rail_bbox: BBox | None = None
    rule_bbox: BBox | None = None
    title_origin: tuple[int, int] = (0, 0)
    subtitle_origin: tuple[int, int] = (0, 0)
    title_color: RGB = (0, 0, 0)
    subtitle_color: RGB = (0, 0, 0)

    def __post_init__(self) -> None:
        if self.template_id not in DEFAULT_TEMPLATES:
            raise ValueError(f"Unknown template ID: {self.template_id}")
        expected_family = "AAD" if self.template_id.startswith("AAD-") else "PAN"
        if self.family != expected_family:
            raise ValueError("Template family does not match template ID")
        if not self.title or not self.subtitle:
            raise ValueError("Template title and subtitle must be non-empty")
        if {field.key for field in self.fields} != set(FIELD_KEYS):
            raise ValueError("Template must define every required field exactly once")
        if len(self.fields) != len(FIELD_KEYS):
            raise ValueError("Template must not define duplicate fields")

        for color in (self.background, self.panel, self.accent, self.text, self.muted):
            _validate_color(color)
        _validate_color(self.title_color)
        _validate_color(self.subtitle_color)
        _validate_bbox(self.panel_bbox)
        _validate_bbox(self.avatar_bbox)
        if self.header_bbox is not None:
            _validate_bbox(self.header_bbox)
        if self.side_rail_bbox is not None:
            _validate_bbox(self.side_rail_bbox)
        if self.rule_bbox is not None:
            _validate_bbox(self.rule_bbox)

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        return {field.key: field for field in self.fields}


def _validate_color(color: RGB) -> None:
    if len(color) != 3 or any(
        not isinstance(channel, int)
        or isinstance(channel, bool)
        or channel < 0
        or channel > 255
        for channel in color
    ):
        raise ValueError(f"Invalid RGB color: {color!r}")


def _validate_bbox(box: BBox) -> None:
    if len(box) != 4:
        raise ValueError(f"Invalid bounding box: {box!r}")
    left, top, width, height = box
    for value in box:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Bounding box values must be integers: {box!r}")
    if left < 0 or top < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Bounding box is outside the canvas: {box!r}")
    if left + width > CANVAS_SIZE[0] or top + height > CANVAS_SIZE[1]:
        raise ValueError(f"Bounding box is outside the canvas: {box!r}")


TEMPLATES: dict[str, TemplateSpec] = {
    "AAD-1": TemplateSpec(
        template_id="AAD-1",
        family="AAD",
        title="FICTIONAL AADHAAR-STYLE SPECIMEN",
        subtitle="Synthetic research layout · Template AAD-1",
        background=(238, 231, 216),
        panel=(252, 249, 242),
        accent=(35, 92, 94),
        text=(35, 40, 42),
        muted=(102, 107, 105),
        header_bbox=(0, 0, 1000, 118),
        panel_bbox=(36, 132, 928, 416),
        avatar_bbox=(60, 164, 190, 240),
        title_origin=(52, 28),
        subtitle_origin=(54, 70),
        title_color=(248, 250, 249),
        subtitle_color=(216, 235, 232),
        fields=(
            FieldSpec("identifier", "IDENTIFIER", (290, 158, 620, 42), 30, bold=True),
            FieldSpec("full_name", "FULL NAME", (290, 240, 620, 46), 34, bold=True),
            FieldSpec("date_of_birth", "DATE OF BIRTH", (290, 326, 620, 40), 28),
            FieldSpec("address", "ADDRESS", (290, 412, 620, 96), 22, max_lines=3),
        ),
    ),
    "PAN-1": TemplateSpec(
        template_id="PAN-1",
        family="PAN",
        title="FICTIONAL PAN-STYLE SPECIMEN",
        subtitle="Synthetic research layout · Template PAN-1",
        background=(244, 236, 224),
        panel=(255, 252, 246),
        accent=(133, 79, 36),
        text=(52, 38, 26),
        muted=(118, 99, 81),
        side_rail_bbox=(0, 0, 20, 625),
        panel_bbox=(44, 36, 912, 512),
        avatar_bbox=(760, 170, 180, 240),
        title_origin=(76, 44),
        subtitle_origin=(78, 86),
        title_color=(133, 79, 36),
        subtitle_color=(118, 99, 81),
        rule_bbox=(76, 120, 868, 3),
        fields=(
            FieldSpec("identifier", "IDENTIFIER", (76, 158, 650, 42), 30, bold=True),
            FieldSpec("full_name", "FULL NAME", (76, 244, 650, 46), 34, bold=True),
            FieldSpec("date_of_birth", "DATE OF BIRTH", (76, 330, 650, 40), 28),
            FieldSpec("address", "ADDRESS", (76, 416, 650, 96), 22, max_lines=3),
        ),
    ),
}


def get_template(template_id: str) -> TemplateSpec:
    try:
        return TEMPLATES[template_id]
    except KeyError as error:
        raise ValueError(f"Unknown template: {template_id}") from error


__all__ = [
    "BBox",
    "CANVAS_SIZE",
    "FIELD_KEYS",
    "FIELD_LABELS",
    "FieldSpec",
    "RGB",
    "TEMPLATES",
    "TemplateSpec",
    "get_template",
]
