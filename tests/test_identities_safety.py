from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.config import load_dataset_config  # noqa: E402
from docdet.data.identities import (  # noqa: E402
    RECOGNIZED_PAN_CATEGORIES,
    generate_identities,
    load_identities,
    save_identities,
    validate_identities,
    verhoeff_is_valid,
)


@pytest.fixture(scope="module")
def dataset_config():
    return load_dataset_config(PROJECT_ROOT / "configs" / "dataset.json")


@pytest.fixture(scope="module")
def identities(dataset_config):
    return generate_identities(dataset_config)


def test_generates_400_identities_with_200_per_template(identities):
    assert len(identities) == 400
    assert Counter(identity.template_id for identity in identities) == {
        "AAD-1": 200,
        "PAN-1": 200,
    }


def test_identity_and_identifier_uniqueness(identities):
    assert len({identity.identity_id for identity in identities}) == 400
    assert len({identity.group_id for identity in identities}) == 400
    assert len({identity.identifier for identity in identities}) == 400


def test_required_identity_fields_are_present(identities):
    for identity in identities:
        assert identity.group_id
        assert identity.template_id in {"AAD-1", "PAN-1"}
        assert identity.template_family in {"AAD", "PAN"}
        assert identity.avatar_seed >= 0
        assert identity.specimen is True
        assert identity.specimen_label == "SPECIMEN – NOT A VALID DOCUMENT"
        assert set(identity.fields) >= {"full_name", "date_of_birth", "address"}
        assert all(value for value in identity.fields.values())


def test_aadhaar_style_identifiers_are_invalid(identities):
    aadhaar = [identity for identity in identities if identity.template_family == "AAD"]
    assert len(aadhaar) == 200
    for identity in aadhaar:
        assert re.fullmatch(r"[01]\d{11}", identity.identifier)
        assert verhoeff_is_valid(identity.identifier) is False


def test_pan_style_identifiers_have_invalid_category(identities):
    pan = [identity for identity in identities if identity.template_family == "PAN"]
    assert len(pan) == 200
    for identity in pan:
        assert re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", identity.identifier)
        assert identity.identifier[3] not in RECOGNIZED_PAN_CATEGORIES


def test_identities_use_only_fictional_fields(identities):
    for identity in identities:
        assert identity.fields["full_name"].startswith("Sample Name ")
        assert "Fictional" in identity.fields["address"]
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", identity.fields["date_of_birth"])


def test_identity_generation_is_deterministic(dataset_config):
    first = [identity.to_dict() for identity in generate_identities(dataset_config)]
    second = [identity.to_dict() for identity in generate_identities(dataset_config)]
    assert first == second


def test_verhoeff_known_values():
    assert verhoeff_is_valid("2363") is True
    assert verhoeff_is_valid("2364") is False


def test_validate_identities_rejects_duplicate_identifiers(identities):
    duplicate = replace(identities[0], identity_id="duplicate-identity")
    with pytest.raises(ValueError, match="identifier"):
        validate_identities([identities[0], duplicate])


def test_save_and_load_identities(tmp_path, identities):
    output_path = tmp_path / "identities.json"
    save_identities(identities, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["count"] == 400
    assert payload["identities"][0] == identities[0].to_dict()

    assert load_identities(output_path) == [
        identity.to_dict() for identity in identities
    ]
