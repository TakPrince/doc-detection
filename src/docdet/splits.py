from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from docdet.config import (
    DEFAULT_TEMPLATES,
    SplitConfig,
    load_split_config,
    resolve_project_path,
)
from docdet.data.identities import load_identities
from docdet.seeding import make_rng

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitAssignment:
    identity_id: str
    group_id: str
    template_id: str
    split: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _value(record: Mapping[str, Any] | object, key: str) -> Any:
    if isinstance(record, Mapping):
        return record[key]
    return getattr(record, key)


def _ratio_counts(size: int, config: SplitConfig) -> tuple[int, int, int]:
    ratios = (config.train_ratio, config.validation_ratio, config.test_ratio)
    raw = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = size - sum(counts)
    priority = sorted(
        range(len(counts)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in priority[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def assign_splits(
    identities: Sequence[Mapping[str, Any] | object],
    config: SplitConfig,
) -> list[SplitAssignment]:
    groups: dict[str, dict[str, Any]] = {}
    for record in identities:
        group_id = str(_value(record, "group_id"))
        template_id = str(_value(record, "template_id"))
        if template_id not in DEFAULT_TEMPLATES:
            raise ValueError(f"Unknown template_id: {template_id}")
        if group_id in groups:
            if groups[group_id]["template_id"] != template_id:
                raise ValueError(f"Group spans multiple templates: {group_id}")
        else:
            groups[group_id] = {"template_id": template_id, "records": []}
        groups[group_id]["records"].append(record)

    assignments: list[SplitAssignment] = []
    for template_id in DEFAULT_TEMPLATES:
        group_ids = sorted(
            group_id
            for group_id, group in groups.items()
            if group["template_id"] == template_id
        )
        rng = make_rng(config.seed, "split", config.split_name, template_id)
        rng.shuffle(group_ids)

        train_count, validation_count, _ = _ratio_counts(len(group_ids), config)
        split_by_group = {group_id: "train" for group_id in group_ids[:train_count]}
        validation_end = train_count + validation_count
        split_by_group.update(
            {
                group_id: "validation"
                for group_id in group_ids[train_count:validation_end]
            }
        )
        split_by_group.update(
            {group_id: "test" for group_id in group_ids[validation_end:]}
        )

        for group_id in group_ids:
            for record in groups[group_id]["records"]:
                assignments.append(
                    SplitAssignment(
                        identity_id=str(_value(record, "identity_id")),
                        group_id=group_id,
                        template_id=template_id,
                        split=split_by_group[group_id],
                    )
                )

    return sorted(assignments, key=lambda assignment: assignment.identity_id)


def validate_splits(
    assignments: Sequence[SplitAssignment],
    identities: Sequence[Mapping[str, Any] | object],
) -> None:
    expected_templates = {
        str(_value(identity, "identity_id")): str(_value(identity, "template_id"))
        for identity in identities
    }
    if len(expected_templates) != len(identities):
        raise ValueError("Identity records contain duplicate identity_id values")

    seen_identities: set[str] = set()
    group_splits: dict[str, set[str]] = {}
    for assignment in assignments:
        if assignment.split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split: {assignment.split}")
        if assignment.identity_id in seen_identities:
            raise ValueError(
                f"Identity assigned more than once: {assignment.identity_id}"
            )
        expected_template = expected_templates.get(assignment.identity_id)
        if expected_template is None:
            raise ValueError(
                f"Unknown identity in split assignment: {assignment.identity_id}"
            )
        if assignment.template_id != expected_template:
            raise ValueError(
                f"Template mismatch for identity: {assignment.identity_id}"
            )
        seen_identities.add(assignment.identity_id)
        group_splits.setdefault(assignment.group_id, set()).add(assignment.split)

    missing = set(expected_templates) - seen_identities
    if missing:
        raise ValueError(
            f"Identities missing from split assignments: {sorted(missing)}"
        )
    leaking = {
        group_id: splits for group_id, splits in group_splits.items() if len(splits) > 1
    }
    if leaking:
        raise ValueError(f"Groups span multiple splits: {sorted(leaking)}")


def save_splits(
    assignments: Sequence[SplitAssignment], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["identity_id", "group_id", "template_id", "split"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(assignment.to_dict())
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Phase 1 grouped splits")
    parser.add_argument("--identities", default="data/identities/identities.json")
    parser.add_argument("--config", default="configs/split_A.json")
    args = parser.parse_args(argv)

    identities = load_identities(resolve_project_path(args.identities))
    config = load_split_config(args.config)
    assignments = assign_splits(identities, config)
    validate_splits(assignments, identities)
    output_path = save_splits(assignments, config.output_path)
    counts = Counter(assignment.split for assignment in assignments)
    print(
        f"Wrote {len(assignments)} split assignments to {output_path} "
        f"(train={counts['train']}, validation={counts['validation']}, "
        f"test={counts['test']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SPLIT_NAMES",
    "SplitAssignment",
    "assign_splits",
    "save_splits",
    "validate_splits",
]
