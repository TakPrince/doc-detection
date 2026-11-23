"""Deterministic, label-independent JPEG degradation for matched pairs."""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass

from PIL import Image

from docdet.seeding import derive_seed

DIFFICULTIES = ("EASY", "MEDIUM")
QUALITY_RANGES = {"EASY": (90, 95), "MEDIUM": (82, 88)}
SUBSAMPLING_OPTIONS = (0, 1)


@dataclass(frozen=True)
class DegradationParams:
    jpeg_quality: int
    subsampling: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if self.subsampling not in (0, 1, 2):
            raise ValueError("subsampling must be 0, 1, or 2")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def sample_degradation(
    base_seed: int,
    identity_id: str,
    pair_id: str,
    difficulty: str,
) -> DegradationParams:
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    rng = random.Random(
        derive_seed(base_seed, "degradation", identity_id, pair_id, difficulty)
    )
    low, high = QUALITY_RANGES[difficulty]
    return DegradationParams(
        jpeg_quality=rng.randint(low, high),
        subsampling=rng.choice(SUBSAMPLING_OPTIONS),
    )


def apply_degradation(image: Image.Image, params: DegradationParams) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=params.jpeg_quality,
        subsampling=params.subsampling,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as degraded:
        return degraded.convert("RGB")


__all__ = ["DegradationParams", "apply_degradation", "sample_degradation"]
