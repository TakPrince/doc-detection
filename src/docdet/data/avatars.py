"""Deterministic procedural avatars built only from Pillow primitives."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

AVATAR_SIZE = (180, 240)

_SKIN_TONES = (
    (246, 214, 188),
    (236, 194, 157),
    (224, 176, 135),
    (205, 150, 110),
    (181, 124, 88),
    (148, 96, 67),
    (112, 73, 52),
)
_HAIR_COLORS = (
    (31, 27, 26),
    (55, 39, 29),
    (82, 55, 35),
    (119, 79, 46),
    (158, 112, 70),
    (194, 190, 184),
    (221, 215, 202),
)
_CLOTHING_COLORS = (
    (53, 91, 112),
    (92, 76, 125),
    (118, 74, 66),
    (60, 106, 87),
    (126, 101, 51),
    (73, 84, 101),
    (136, 92, 111),
)
_BACKGROUNDS = (
    (221, 232, 229),
    (231, 224, 236),
    (234, 228, 213),
    (219, 226, 239),
    (236, 220, 216),
    (224, 235, 218),
    (232, 231, 226),
)
_ACCENTS = (
    (72, 119, 125),
    (122, 93, 137),
    (158, 102, 80),
    (83, 132, 103),
    (164, 128, 70),
    (84, 111, 145),
    (150, 105, 122),
)


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("avatar seed must be an integer")
    if seed < 0:
        raise ValueError("avatar seed must be non-negative")


def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("avatar size must be a (width, height) tuple")
    width, height = size
    for value in (width, height):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("avatar size values must be positive integers")
    return width, height


def _ellipse_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[int, int, int, int]:
    left = round(center_x - width / 2)
    top = round(center_y - height / 2)
    right = round(center_x + width / 2)
    bottom = round(center_y + height / 2)
    return left, top, right, bottom


def render_avatar(seed: int, size: tuple[int, int] = AVATAR_SIZE) -> Image.Image:
    _validate_seed(seed)
    width, height = _validate_size(size)
    rng = random.Random(seed)

    background = rng.choice(_BACKGROUNDS)
    accent = rng.choice(_ACCENTS)
    skin = rng.choice(_SKIN_TONES)
    hair = rng.choice(_HAIR_COLORS)
    clothing = rng.choice(_CLOTHING_COLORS)

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    for _ in range(rng.randint(2, 4)):
        radius = rng.uniform(0.12, 0.28) * width
        center_x = rng.uniform(0.05, 0.95) * width
        center_y = rng.uniform(0.05, 0.95) * height
        draw.ellipse(
            _ellipse_box(center_x, center_y, radius * 2, radius * 2),
            fill=accent,
        )

    center_x = width / 2 + rng.uniform(-0.05, 0.05) * width
    head_width = rng.uniform(0.52, 0.61) * width
    head_height = rng.uniform(0.52, 0.60) * height
    head_top = rng.uniform(0.12, 0.17) * height
    head_center_y = head_top + head_height / 2

    shoulder_top = head_top + head_height * rng.uniform(0.78, 0.92)
    shoulder_width = rng.uniform(1.22, 1.48) * width
    draw.ellipse(
        _ellipse_box(
            center_x,
            shoulder_top + height * 0.22,
            shoulder_width,
            height * 0.62,
        ),
        fill=clothing,
    )

    neck_width = rng.uniform(0.16, 0.22) * width
    draw.rounded_rectangle(
        (
            round(center_x - neck_width / 2),
            round(head_top + head_height * 0.68),
            round(center_x + neck_width / 2),
            round(shoulder_top + height * 0.05),
        ),
        radius=max(2, round(neck_width * 0.35)),
        fill=skin,
    )

    hair_width = head_width * rng.uniform(1.08, 1.20)
    hair_height = head_height * rng.uniform(1.06, 1.16)
    hair_top = head_top - height * rng.uniform(0.02, 0.05)
    draw.ellipse(
        _ellipse_box(
            center_x,
            hair_top + hair_height / 2,
            hair_width,
            hair_height,
        ),
        fill=hair,
    )

    long_hair = rng.random() < 0.38
    if long_hair:
        side_width = hair_width * rng.uniform(0.18, 0.26)
        for direction in (-1, 1):
            side_x = center_x + direction * hair_width * 0.42
            draw.rounded_rectangle(
                (
                    round(side_x - side_width / 2),
                    round(head_top + head_height * 0.28),
                    round(side_x + side_width / 2),
                    round(shoulder_top + height * 0.12),
                ),
                radius=max(2, round(side_width * 0.4)),
                fill=hair,
            )

    face_width = head_width * rng.uniform(0.88, 0.96)
    face_height = head_height * rng.uniform(0.90, 0.98)
    face_center_y = head_top + face_height / 2 + height * 0.03
    face_box = _ellipse_box(center_x, face_center_y, face_width, face_height)
    draw.ellipse(face_box, fill=skin)

    ear_width = face_width * 0.14
    ear_height = face_height * 0.18
    for direction in (-1, 1):
        draw.ellipse(
            _ellipse_box(
                center_x + direction * face_width * 0.51,
                face_center_y + face_height * 0.02,
                ear_width,
                ear_height,
            ),
            fill=skin,
            outline=_darken(skin, 0.82),
            width=max(1, width // 90),
        )

    fringe_style = rng.randrange(3)
    if fringe_style == 0:
        fringe_box = (
            round(center_x - face_width * 0.44),
            round(head_top + face_height * 0.01),
            round(center_x + face_width * 0.36),
            round(head_top + face_height * 0.28),
        )
        draw.chord(fringe_box, 180, 360, fill=hair)
    elif fringe_style == 1:
        fringe_box = (
            round(center_x - face_width * 0.30),
            round(head_top + face_height * 0.00),
            round(center_x + face_width * 0.48),
            round(head_top + face_height * 0.24),
        )
        draw.chord(fringe_box, 180, 360, fill=hair)
    else:
        sweep_box = (
            round(center_x - face_width * 0.48),
            round(head_top + face_height * 0.04),
            round(center_x + face_width * 0.22),
            round(head_top + face_height * 0.30),
        )
        draw.chord(sweep_box, 150, 345, fill=hair)

    eye_y = face_center_y - face_height * 0.08
    eye_spacing = face_width * rng.uniform(0.20, 0.25)
    eye_width = face_width * rng.uniform(0.12, 0.16)
    eye_height = face_height * rng.uniform(0.045, 0.060)
    pupil_radius = max(2, round(min(eye_width, eye_height) * 0.36))
    for direction in (-1, 1):
        eye_center_x = center_x + direction * eye_spacing
        draw.ellipse(
            _ellipse_box(eye_center_x, eye_y, eye_width, eye_height),
            fill=(250, 249, 246),
            outline=_darken(skin, 0.68),
            width=max(1, width // 120),
        )
        pupil_x = eye_center_x + rng.uniform(-0.12, 0.12) * eye_width
        pupil_y = eye_y + rng.uniform(-0.08, 0.08) * eye_height
        draw.ellipse(
            _ellipse_box(pupil_x, pupil_y, pupil_radius * 2, pupil_radius * 2),
            fill=(38, 33, 31),
        )

    brow_y = eye_y - eye_height * rng.uniform(1.4, 1.9)
    for direction in (-1, 1):
        brow_center_x = center_x + direction * eye_spacing
        draw.line(
            (
                round(brow_center_x - eye_width * 0.55),
                round(brow_y + rng.uniform(-1.0, 1.0)),
                round(brow_center_x + eye_width * 0.55),
                round(brow_y + rng.uniform(-1.0, 1.0)),
            ),
            fill=_darken(hair, 0.8),
            width=max(2, round(height * 0.018)),
        )

    if rng.random() < 0.32:
        glasses_box_width = eye_spacing * 2 + eye_width * 1.8
        glasses_box_height = eye_height * 2.8
        draw.rounded_rectangle(
            (
                round(center_x - glasses_box_width / 2),
                round(eye_y - glasses_box_height / 2),
                round(center_x + glasses_box_width / 2),
                round(eye_y + glasses_box_height / 2),
            ),
            radius=max(3, round(height * 0.02)),
            outline=_darken(clothing, 0.65),
            width=max(2, round(width // 80)),
        )
        draw.line(
            (
                round(center_x - eye_spacing + eye_width / 2),
                round(eye_y),
                round(center_x + eye_spacing - eye_width / 2),
                round(eye_y),
            ),
            fill=_darken(clothing, 0.65),
            width=max(2, round(width // 90)),
        )

    nose_top = eye_y + eye_height * 0.75
    nose_bottom = face_center_y + face_height * 0.16
    draw.line(
        (
            round(center_x + rng.uniform(-0.02, 0.02) * face_width),
            round(nose_top),
            round(center_x + rng.uniform(-0.03, 0.03) * face_width),
            round(nose_bottom),
        ),
        fill=_darken(skin, 0.78),
        width=max(1, round(width // 100)),
    )

    mouth_y = face_center_y + face_height * rng.uniform(0.26, 0.31)
    mouth_width = face_width * rng.uniform(0.24, 0.32)
    smile = rng.random() < 0.68
    mouth_box = (
        round(center_x - mouth_width / 2),
        round(mouth_y - face_height * 0.05),
        round(center_x + mouth_width / 2),
        round(mouth_y + face_height * 0.06),
    )
    if smile:
        draw.arc(
            mouth_box,
            10,
            170,
            fill=(118, 64, 61),
            width=max(2, round(height * 0.012)),
        )
    else:
        draw.line(
            (
                round(center_x - mouth_width / 2),
                round(mouth_y),
                round(center_x + mouth_width / 2),
                round(mouth_y),
            ),
            fill=(118, 64, 61),
            width=max(2, round(height * 0.012)),
        )

    collar_left = round(center_x - shoulder_width * 0.16)
    collar_right = round(center_x + shoulder_width * 0.16)
    collar_y = round(shoulder_top + height * 0.035)
    draw.polygon(
        (
            (collar_left, collar_y),
            (round(center_x), round(collar_y + height * 0.12)),
            (collar_right, collar_y),
            (round(center_x), round(collar_y + height * 0.18)),
        ),
        fill=_lighten(clothing, 1.18),
    )

    return image


def _darken(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


def _lighten(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


__all__ = ["AVATAR_SIZE", "render_avatar"]
