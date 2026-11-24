from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATES = ("AAD-1", "PAN-1")
SPECIMEN_LABEL = "SPECIMEN – NOT A VALID DOCUMENT"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return payload


@dataclass(frozen=True)
class DatasetConfig:
    seed: int
    total_identities: int
    identities_per_template: int
    templates: tuple[str, ...]
    output_path: Path
    specimen_label: str = SPECIMEN_LABEL

    def __post_init__(self) -> None:
        templates = tuple(self.templates)
        object.__setattr__(self, "templates", templates)

        if not isinstance(self.seed, int):
            raise ValueError("Dataset seed must be an integer")
        if self.identities_per_template <= 0:
            raise ValueError("identities_per_template must be positive")
        if templates != DEFAULT_TEMPLATES:
            raise ValueError(f"templates must be exactly {DEFAULT_TEMPLATES}")
        expected_total = self.identities_per_template * len(templates)
        if self.total_identities != expected_total:
            raise ValueError(
                "total_identities must equal identities_per_template * len(templates)"
            )
        if self.specimen_label != SPECIMEN_LABEL:
            raise ValueError(f"specimen_label must be {SPECIMEN_LABEL!r}")
        object.__setattr__(self, "output_path", resolve_project_path(self.output_path))


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    output_path: Path
    split_name: str = "A"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("Split seed must be an integer")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("Split ratios must be positive")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")
        if not isinstance(self.split_name, str) or not self.split_name:
            raise ValueError("split_name must be a non-empty string")
        object.__setattr__(self, "output_path", resolve_project_path(self.output_path))


def load_dataset_config(path: str | Path) -> DatasetConfig:
    payload = load_json_config(path)
    return DatasetConfig(
        seed=payload["seed"],
        total_identities=payload["total_identities"],
        identities_per_template=payload["identities_per_template"],
        templates=tuple(payload["templates"]),
        output_path=Path(payload["output_path"]),
        specimen_label=payload.get("specimen_label", SPECIMEN_LABEL),
    )


def load_split_config(path: str | Path) -> SplitConfig:
    payload = load_json_config(path)
    return SplitConfig(
        seed=payload["seed"],
        train_ratio=float(payload["train_ratio"]),
        validation_ratio=float(payload["validation_ratio"]),
        test_ratio=float(payload["test_ratio"]),
        output_path=Path(payload["output_path"]),
        split_name=str(payload.get("split_name", "A")),
    )


__all__ = [
    "DEFAULT_TEMPLATES",
    "PROJECT_ROOT",
    "SPECIMEN_LABEL",
    "DatasetConfig",
    "SplitConfig",
    "load_dataset_config",
    "load_json_config",
    "load_split_config",
    "resolve_project_path",
]
