from __future__ import annotations

import argparse
import json
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from docdet.config import (
    DEFAULT_TEMPLATES,
    SPECIMEN_LABEL,
    DatasetConfig,
    load_dataset_config,
)
from docdet.seeding import make_rng

RECOGNIZED_PAN_CATEGORIES = frozenset("PCHFATBLJG")
INVALID_PAN_CATEGORY_LETTERS = tuple(
    sorted(set(string.ascii_uppercase) - RECOGNIZED_PAN_CATEGORIES)
)
AADHAAR_PATTERN = re.compile(r"[01]\d{11}")
PAN_PATTERN = re.compile(r"[A-Z]{5}\d{4}[A-Z]")
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


@dataclass(frozen=True)
class Identity:
    identity_id: str
    group_id: str
    template_id: str
    template_family: str
    identifier: str
    identifier_style: str
    fields: dict[str, str]
    avatar_seed: int
    specimen: bool
    specimen_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verhoeff_is_valid(number: str) -> bool:
    if not isinstance(number, str) or not number.isdigit():
        return False
    checksum = 0
    for index, character in enumerate(reversed(number)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(character)]]
    return checksum == 0


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


def _generate_aadhaar_style_identifier(rng, used: set[str]) -> str:
    for _ in range(10_000):
        candidate = rng.choice(("0", "1")) + "".join(
            str(rng.randrange(10)) for _ in range(11)
        )
        if candidate not in used and not verhoeff_is_valid(candidate):
            used.add(candidate)
            return candidate
    raise RuntimeError("Could not generate a unique invalid Aadhaar-style identifier")


def _generate_pan_style_identifier(rng, used: set[str]) -> str:
    for _ in range(10_000):
        candidate = (
            "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
            + rng.choice(INVALID_PAN_CATEGORY_LETTERS)
            + rng.choice(string.ascii_uppercase)
            + "".join(str(rng.randrange(10)) for _ in range(4))
            + rng.choice(string.ascii_uppercase)
        )
        if candidate not in used and candidate[3] not in RECOGNIZED_PAN_CATEGORIES:
            used.add(candidate)
            return candidate
    raise RuntimeError("Could not generate a unique invalid PAN-style identifier")


def _fictional_fields(rng, *, sequence: int) -> dict[str, str]:
    year = rng.randrange(1970, 2001)
    month = rng.randrange(1, 13)
    day = rng.randrange(1, 29)
    return {
        "full_name": f"Sample Name {sequence:04d}",
        "date_of_birth": date(year, month, day).isoformat(),
        "address": (
            f"{rng.randrange(1, 1000)} Fictional Street, "
            f"Demo District, Demo State {rng.randrange(100000, 1000000)}"
        ),
    }


def generate_identities(config: DatasetConfig) -> list[Identity]:
    identities: list[Identity] = []
    used_identifiers: set[str] = set()
    sequence = 1

    for template_id in config.templates:
        family = "AAD" if template_id.startswith("AAD-") else "PAN"
        for index in range(1, config.identities_per_template + 1):
            rng = make_rng(config.seed, "identity", template_id, index)
            identity_id = f"{template_id}-{index:04d}"
            group_id = f"grp-{identity_id}"

            if family == "AAD":
                identifier = _generate_aadhaar_style_identifier(rng, used_identifiers)
                identifier_style = "AADHAAR_STYLE"
            else:
                identifier = _generate_pan_style_identifier(rng, used_identifiers)
                identifier_style = "PAN_STYLE"

            identities.append(
                Identity(
                    identity_id=identity_id,
                    group_id=group_id,
                    template_id=template_id,
                    template_family=family,
                    identifier=identifier,
                    identifier_style=identifier_style,
                    fields=_fictional_fields(rng, sequence=sequence),
                    avatar_seed=rng.randrange(1, 2**31),
                    specimen=True,
                    specimen_label=config.specimen_label,
                )
            )
            sequence += 1

    validate_identities(identities)
    return identities


def validate_identity(identity: Identity) -> None:
    if not identity.identity_id or not identity.group_id:
        raise ValueError("Identity and group IDs must be non-empty")
    if identity.template_id not in DEFAULT_TEMPLATES:
        raise ValueError(f"Unknown template: {identity.template_id}")
    expected_family = "AAD" if identity.template_id.startswith("AAD-") else "PAN"
    if identity.template_family != expected_family:
        raise ValueError("template_family does not match template_id")
    if identity.specimen is not True:
        raise ValueError("Every identity must be flagged as a specimen")
    if identity.specimen_label != SPECIMEN_LABEL:
        raise ValueError("Specimen label does not match the required safety label")
    if not isinstance(identity.avatar_seed, int) or isinstance(
        identity.avatar_seed, bool
    ):
        raise ValueError("avatar_seed must be an integer")
    if identity.avatar_seed < 0:
        raise ValueError("avatar_seed must be non-negative")

    fields = identity.fields
    if not isinstance(fields, Mapping):
        raise ValueError("Identity fields must be a mapping")
    for field_name in ("full_name", "date_of_birth", "address"):
        value = fields.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing fictional field: {field_name}")
    if not fields["full_name"].startswith("Sample Name "):
        raise ValueError("full_name must use the synthetic sample format")
    if "Fictional" not in fields["address"]:
        raise ValueError("address must be clearly fictional")
    if not DATE_PATTERN.fullmatch(fields["date_of_birth"]):
        raise ValueError("date_of_birth must use ISO format")
    try:
        date.fromisoformat(fields["date_of_birth"])
    except ValueError as error:
        raise ValueError("date_of_birth must be a valid date") from error

    if identity.template_family == "AAD":
        if identity.identifier_style != "AADHAAR_STYLE":
            raise ValueError("Aadhaar-style identifier_style mismatch")
        if not AADHAAR_PATTERN.fullmatch(identity.identifier):
            raise ValueError(
                "Aadhaar-style identifier must be 12 digits starting 0 or 1"
            )
        if verhoeff_is_valid(identity.identifier):
            raise ValueError("Aadhaar-style identifier must fail Verhoeff validation")
    else:
        if identity.identifier_style != "PAN_STYLE":
            raise ValueError("PAN-style identifier_style mismatch")
        if not PAN_PATTERN.fullmatch(identity.identifier):
            raise ValueError("PAN-style identifier has an invalid shape")
        if identity.identifier[3] in RECOGNIZED_PAN_CATEGORIES:
            raise ValueError(
                "PAN-style identifier must use an invalid category character"
            )


def validate_identities(identities: Sequence[Identity]) -> None:
    seen_ids: set[str] = set()
    seen_identifiers: set[str] = set()
    for identity in identities:
        validate_identity(identity)
        if identity.identity_id in seen_ids:
            raise ValueError(f"Duplicate identity_id: {identity.identity_id}")
        if identity.identifier in seen_identifiers:
            raise ValueError(f"Duplicate identifier: {identity.identifier}")
        seen_ids.add(identity.identity_id)
        seen_identifiers.add(identity.identifier)


def save_identities(identities: Sequence[Identity], output_path: str | Path) -> Path:
    validate_identities(identities)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "count": len(identities),
        "identities": [identity.to_dict() for identity in identities],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_identities(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("identities")
    else:
        records = payload
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError("Identity file must contain a list of identity objects")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Phase 1 identities"
    )
    parser.add_argument("--config", default="configs/dataset.json")
    args = parser.parse_args(argv)

    config = load_dataset_config(args.config)
    identities = generate_identities(config)
    output_path = save_identities(identities, config.output_path)
    print(f"Wrote {len(identities)} identities to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AADHAAR_PATTERN",
    "INVALID_PAN_CATEGORY_LETTERS",
    "PAN_PATTERN",
    "RECOGNIZED_PAN_CATEGORIES",
    "Identity",
    "generate_identities",
    "identifier_is_invalid",
    "load_identities",
    "save_identities",
    "validate_identities",
    "validate_identity",
    "verhoeff_is_valid",
]
