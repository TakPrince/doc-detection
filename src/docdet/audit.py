"""Machine-readable integrity audit for the generated specimen dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from docdet.config import resolve_project_path
from docdet.data.tamper import (
    DIFFICULTIES,
    TAMPER_CATEGORIES,
    identifier_is_invalid,
)

DEFAULT_MANIFEST_PATH = "data/manifests/dataset.jsonl"
DEFAULT_DATA_ROOT = "data/generated"
DEFAULT_SPLITS_PATH = "data/splits/splits_A.csv"
DEFAULT_REPORT_PATH = "results/audit.json"
DEFAULT_EXPECTED_COUNTS = {
    "images": 2400,
    "identities": 400,
    "pairs": 1200,
    "label_counts": {"GENUINE": 1200, "TAMPERED": 1200},
    "category_counts": {category: 800 for category in TAMPER_CATEGORIES},
    "difficulty_counts": {"EASY": 1200, "MEDIUM": 1200},
    "template_counts": {"AAD-1": 600, "AAD-2": 600, "PAN-1": 600, "PAN-2": 600},
    "split_counts": {"train": 1680, "validation": 360, "test": 360},
}
REQUIRED_FIELDS = {
    "image_id",
    "pair_id",
    "group_id",
    "identity_id",
    "template_id",
    "template_family",
    "split",
    "label",
    "category",
    "difficulty",
    "target_field",
    "image_path",
    "mask_path",
    "operation",
    "identifier",
    "original_identifier",
    "pair_seed",
    "tamper_seed",
    "degradation_seed",
    "degradation_params",
    "capture_settings",
    "specimen_label",
}
SHARED_PAIR_FIELDS = (
    "identity_id",
    "group_id",
    "template_id",
    "split",
    "category",
    "difficulty",
    "target_field",
    "base_render_id",
    "base_render_seed",
    "pair_seed",
    "tamper_seed",
    "degradation_seed",
    "degradation_params",
    "capture_settings",
)


class AuditError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class AuditConfig:
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH
    data_root: str | Path = DEFAULT_DATA_ROOT
    splits_path: str | Path | None = DEFAULT_SPLITS_PATH
    report_path: str | Path | None = DEFAULT_REPORT_PATH
    expected_counts: Mapping[str, Any] | None = None
    image_size: tuple[int, int] = (1000, 625)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "manifest_path", resolve_project_path(self.manifest_path)
        )
        object.__setattr__(self, "data_root", resolve_project_path(self.data_root))
        if self.splits_path is not None:
            object.__setattr__(
                self, "splits_path", resolve_project_path(self.splits_path)
            )
        if self.report_path is not None:
            object.__setattr__(
                self, "report_path", resolve_project_path(self.report_path)
            )
        if self.expected_counts is not None:
            object.__setattr__(self, "expected_counts", dict(self.expected_counts))


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    raise ValueError("manifest rows must be JSON objects")
                entries.append(entry)
    return entries


def _load_splits(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["identity_id"]: row["split"] for row in csv.DictReader(handle)}


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _count_signature(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    templates = sorted({row["template_id"] for row in entries if "template_id" in row})
    splits = sorted({row["split"] for row in entries if "split" in row})
    return {
        "images": len(entries),
        "identities": len({row.get("identity_id") for row in entries}),
        "pairs": len({row.get("pair_id") for row in entries}),
        "label_counts": {
            label: sum(row.get("label") == label for row in entries)
            for label in ("GENUINE", "TAMPERED")
        },
        "category_counts": {
            category: sum(row.get("category") == category for row in entries)
            for category in TAMPER_CATEGORIES
        },
        "difficulty_counts": {
            difficulty: sum(row.get("difficulty") == difficulty for row in entries)
            for difficulty in DIFFICULTIES
        },
        "template_counts": {
            template: sum(row.get("template_id") == template for row in entries)
            for template in templates
        },
        "split_counts": {
            split: sum(row.get("split") == split for row in entries)
            for split in splits
        },
    }


def _matches_expected(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    for key, value in expected.items():
        if isinstance(value, Mapping):
            actual_value = actual.get(key)
            if not isinstance(actual_value, Mapping):
                return False
            if any(actual_value.get(name) != expected_value for name, expected_value in value.items()):
                return False
        elif actual.get(key) != value:
            return False
    return True


def run_audit(config: AuditConfig) -> dict[str, Any]:
    entries = _load_manifest(config.manifest_path)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, **details: Any) -> None:
        checks[name] = {"passed": passed, "details": details}

    image_ids = [row.get("image_id") for row in entries]
    path_fields_ok = all(
        _is_safe_relative_path(row.get("image_path"))
        and _is_safe_relative_path(row.get("mask_path"))
        for row in entries
    )
    fields_ok = all(REQUIRED_FIELDS <= set(row) for row in entries)
    record(
        "manifest_integrity",
        fields_ok and path_fields_ok and len(image_ids) == len(set(image_ids)),
        row_count=len(entries),
        missing_fields=sum(bool(REQUIRED_FIELDS - set(row)) for row in entries),
        duplicate_image_ids=len(image_ids) - len(set(image_ids)),
        unsafe_paths=sum(not _is_safe_relative_path(row.get("image_path")) for row in entries),
    )

    file_issues = 0
    for row in entries:
        image_path = config.data_root / row["image_path"]
        mask_path = config.data_root / row["mask_path"]
        if not image_path.is_file() or not mask_path.is_file():
            file_issues += 1
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != config.image_size or image.mode != "RGB":
                file_issues += 1
            if mask.size != config.image_size or mask.mode != "L":
                file_issues += 1
    record("file_integrity", file_issues == 0, checked=len(entries), issues=file_issues)

    actual_counts = _count_signature(entries)
    expected_counts = config.expected_counts or DEFAULT_EXPECTED_COUNTS
    record(
        "balance",
        _matches_expected(actual_counts, expected_counts),
        actual=actual_counts,
        expected=dict(expected_counts),
    )
    record(
        "deterministic_counts",
        _count_signature(entries) == actual_counts
        and actual_counts.get("images") == expected_counts.get("images")
        and actual_counts.get("identities") == expected_counts.get("identities")
        and actual_counts.get("pairs") == expected_counts.get("pairs"),
        signature=actual_counts,
    )

    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in entries:
        pairs[row.get("pair_id", "")].append(row)
    pair_issues = 0
    for members in pairs.values():
        if len(members) != 2:
            pair_issues += 1
            continue
        labels = {member["label"] for member in members}
        if labels != {"GENUINE", "TAMPERED"}:
            pair_issues += 1
        for field in SHARED_PAIR_FIELDS:
            if members[0].get(field) != members[1].get(field):
                pair_issues += 1
    record("pair_integrity", pair_issues == 0, pair_count=len(pairs), issues=pair_issues)

    mask_issues = 0
    for row in entries:
        mask_path = config.data_root / row["mask_path"]
        if not mask_path.is_file():
            mask_issues += 1
            continue
        with Image.open(mask_path) as mask:
            nonempty = mask.getbbox() is not None
        if (row["label"] == "GENUINE" and nonempty) or (
            row["label"] == "TAMPERED" and not nonempty
        ):
            mask_issues += 1
    record("masks", mask_issues == 0, checked=len(entries), issues=mask_issues)

    identity_splits: dict[str, set[str]] = defaultdict(set)
    pair_identities: dict[str, set[str]] = defaultdict(set)
    pair_splits: dict[str, set[str]] = defaultdict(set)
    group_identities: dict[str, set[str]] = defaultdict(set)
    split_issues = 0
    source_splits = _load_splits(config.splits_path)
    for row in entries:
        identity_splits[row["identity_id"]].add(row["split"])
        pair_identities[row["pair_id"]].add(row["identity_id"])
        pair_splits[row["pair_id"]].add(row["split"])
        group_identities[row["group_id"]].add(row["identity_id"])
        if source_splits and source_splits.get(row["identity_id"]) != row["split"]:
            split_issues += 1
    if any(len(values) != 1 for values in identity_splits.values()):
        split_issues += 1
    if any(len(values) != 1 for values in pair_identities.values()):
        split_issues += 1
    if any(len(values) != 1 for values in pair_splits.values()):
        split_issues += 1
    if any(len(values) != 1 for values in group_identities.values()):
        split_issues += 1
    record("leakage", split_issues == 0, issues=split_issues)

    digest_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    duplicate_images = 0
    allowed_null_duplicates = 0
    for row in entries:
        image_path = config.data_root / row["image_path"]
        if not image_path.is_file():
            continue
        digest = _digest(image_path)
        for prior in digest_rows[digest]:
            if (
                row["label"] == "GENUINE"
                and prior["label"] == "GENUINE"
                and row["identity_id"] == prior["identity_id"]
                and row["base_render_id"] == prior["base_render_id"]
            ):
                allowed_null_duplicates += 1
            else:
                duplicate_images += 1
        digest_rows[digest].append(row)
    identifier_identities: dict[str, set[str]] = defaultdict(set)
    for row in entries:
        identifier_identities[row["identifier"]].add(row["identity_id"])
    duplicate_identifiers = sum(
        len(identities) > 1 for identities in identifier_identities.values()
    )
    record(
        "duplicates",
        duplicate_images == 0 and duplicate_identifiers == 0,
        duplicate_images=duplicate_images,
        allowed_null_edit_duplicates=allowed_null_duplicates,
        duplicate_identifiers=duplicate_identifiers,
    )

    invalid_count = sum(
        identifier_is_invalid(row["identifier"], row["template_family"])
        for row in entries
    )
    record(
        "identifier_safety",
        invalid_count == len(entries),
        checked=len(entries),
        invalid=invalid_count,
    )

    failures = [name for name, check in checks.items() if not check["passed"]]
    report = {
        "schema_version": 1,
        "status": "FAIL" if failures else "PASS",
        "summary": {
            "manifest_path": str(config.manifest_path),
            "data_root": str(config.data_root),
            "image_count": len(entries),
            "identity_count": actual_counts["identities"],
            "pair_count": actual_counts["pairs"],
            "failures": failures,
        },
        "checks": checks,
    }
    if config.report_path is not None:
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        config.report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if failures:
        raise AuditError(f"Audit failed: {', '.join(failures)}", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the generated specimen dataset")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--splits", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    config = AuditConfig(
        manifest_path=args.manifest,
        data_root=args.data_root,
        splits_path=args.splits,
        report_path=args.report,
    )
    try:
        report = run_audit(config)
    except AuditError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EXPECTED_COUNTS",
    "AuditConfig",
    "AuditError",
    "run_audit",
]
