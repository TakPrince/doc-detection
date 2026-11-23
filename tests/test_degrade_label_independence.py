from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.data.degrade import (  # noqa: E402
    apply_degradation,
    sample_degradation,
)


def _pair_seed() -> int:
    return 20261124


def test_sampler_has_no_class_label_parameter():
    parameters = inspect.signature(sample_degradation).parameters
    assert "label" not in parameters
    assert "class_label" not in parameters


def test_same_pair_and_difficulty_produce_same_degradation():
    first = sample_degradation(
        _pair_seed(), "AAD-1-0001", "AAD-1-0001__TEXT__EASY", "EASY"
    )
    second = sample_degradation(
        _pair_seed(), "AAD-1-0001", "AAD-1-0001__TEXT__EASY", "EASY"
    )

    assert first == second
    assert 80 <= first.jpeg_quality <= 96
    assert first.subsampling in (0, 1, 2)


def test_difficulty_controls_quality_range():
    easy = {
        sample_degradation(_pair_seed(), f"identity-{index}", f"pair-{index}", "EASY").jpeg_quality
        for index in range(24)
    }
    medium = {
        sample_degradation(
            _pair_seed(), f"identity-{index}", f"pair-{index}", "MEDIUM"
        ).jpeg_quality
        for index in range(24)
    }

    assert easy
    assert medium
    assert min(easy) > min(medium)
    assert max(easy) > max(medium)


def test_degradation_is_deterministic_and_preserves_image_shape():
    image = Image.new("RGB", (240, 150), (123, 45, 67))
    params = sample_degradation(_pair_seed(), "id", "pair", "MEDIUM")
    before = image.tobytes()

    first = apply_degradation(image, params)
    second = apply_degradation(image, params)

    assert first.tobytes() == second.tobytes()
    assert first.size == image.size
    assert first.mode == "RGB"
    assert image.tobytes() == before


def test_degradation_metadata_is_json_serializable():
    params = sample_degradation(_pair_seed(), "id", "pair", "EASY")
    payload = params.to_dict()

    assert payload["jpeg_quality"] == params.jpeg_quality
    assert payload["subsampling"] == params.subsampling
    assert isinstance(payload["schema_version"], int)


def test_degradation_rejects_invalid_difficulty():
    with pytest.raises(ValueError):
        sample_degradation(_pair_seed(), "id", "pair", "HARD")
