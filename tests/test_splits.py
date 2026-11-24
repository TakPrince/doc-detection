from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.config import load_dataset_config, load_split_config  # noqa: E402
from docdet.data.identities import generate_identities  # noqa: E402
from docdet.splits import assign_splits, save_splits, validate_splits  # noqa: E402


@pytest.fixture(scope="module")
def identities():
    return generate_identities(
        load_dataset_config(PROJECT_ROOT / "configs" / "dataset.json")
    )


@pytest.fixture(scope="module")
def split_config():
    return load_split_config(PROJECT_ROOT / "configs" / "split_A.json")


@pytest.fixture(scope="module")
def assignments(identities, split_config):
    return assign_splits(identities, split_config)


def test_split_proportions(assignments):
    assert Counter(assignment.split for assignment in assignments) == {
        "train": 280,
        "validation": 60,
        "test": 60,
    }


def test_template_distribution_is_stratified(assignments):
    counts = Counter(
        (assignment.template_id, assignment.split) for assignment in assignments
    )
    for template_id in ("AAD-1", "PAN-1"):
        assert counts[(template_id, "train")] == 140
        assert counts[(template_id, "validation")] == 30
        assert counts[(template_id, "test")] == 30


def test_no_identity_or_group_leakage(identities, assignments):
    identity_splits = defaultdict(set)
    group_splits = defaultdict(set)
    for assignment in assignments:
        identity_splits[assignment.identity_id].add(assignment.split)
        group_splits[assignment.group_id].add(assignment.split)

    assert len(identity_splits) == 400
    assert all(len(splits) == 1 for splits in identity_splits.values())
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert set(identity_splits) == {identity.identity_id for identity in identities}


def test_split_assignment_is_deterministic(identities, split_config, assignments):
    assert assign_splits(identities, split_config) == assignments


def test_derivatives_of_an_identity_stay_in_one_split(identities, split_config):
    base = identities[0]
    derivative = replace(
        base,
        identity_id=f"{base.identity_id}-derivative",
        avatar_seed=base.avatar_seed + 1,
    )
    derived_assignments = assign_splits([*identities, derivative], split_config)
    splits = {
        assignment.split
        for assignment in derived_assignments
        if assignment.group_id == base.group_id
    }
    assert len(splits) == 1


def test_validate_splits_accepts_valid_assignments(identities, assignments):
    validate_splits(assignments, identities)


def test_save_splits_csv(tmp_path, assignments):
    output_path = tmp_path / "splits_A.csv"
    save_splits(assignments, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["identity_id", "group_id", "template_id", "split"]
    assert len(rows) == 400
    assert {row["split"] for row in rows} == {"train", "validation", "test"}


def test_invalid_split_ratios_are_rejected(tmp_path):
    from docdet.config import SplitConfig

    with pytest.raises(ValueError, match="ratio"):
        SplitConfig(
            seed=1,
            train_ratio=0.70,
            validation_ratio=0.20,
            test_ratio=0.20,
            output_path=tmp_path / "splits.csv",
        )


def test_split_config_loads(split_config):
    assert split_config.split_name == "A"
    assert split_config.train_ratio == pytest.approx(0.70)
    assert split_config.validation_ratio == pytest.approx(0.15)
    assert split_config.test_ratio == pytest.approx(0.15)
    assert split_config.output_path.name == "splits_A.csv"
