from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.audit import AuditConfig, AuditError, run_audit  # noqa: E402
from docdet.data.identities import load_identities  # noqa: E402

IMAGE_SIZE = (64, 40)


def _write_mini_dataset(tmp_path: Path):
    data_root = tmp_path / "data"
    manifest_path = tmp_path / "manifest.jsonl"
    splits_path = tmp_path / "splits.csv"
    report_path = tmp_path / "audit.json"
    identities = load_identities(PROJECT_ROOT / "data" / "identities" / "identities.json")[:4]
    rows = []

    splits_path.write_text("identity_id,group_id,template_id,split\n", encoding="utf-8")
    for index, identity in enumerate(identities):
        split = "train" if index < 2 else "test"
        template_id = identity["template_id"]
        with splits_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{identity['identity_id']},{identity['group_id']},{template_id},{split}\n")

        pair_id = f"{identity['identity_id']}__TEXT"
        for label_index, label in enumerate(("GENUINE", "TAMPERED")):
            image_id = f"{pair_id}__{label}"
            image_path = Path("images") / split / f"{image_id}.png"
            (data_root / image_path).parent.mkdir(parents=True, exist_ok=True)
            color = (index * 20, 30, 40 if label == "GENUINE" else 150)
            Image.new("RGB", IMAGE_SIZE, color).save(data_root / image_path)
            rows.append(
                {
                    "schema_version": 1,
                    "image_id": image_id,
                    "pair_id": pair_id,
                    "group_id": identity["group_id"],
                    "identity_id": identity["identity_id"],
                    "template_id": template_id,
                    "template_family": identity["template_family"],
                    "split": split,
                    "label": label,
                    "category": "TEXT",
                    "target_field": "full_name",
                    "image_path": str(image_path),
                    "operation": "NULL_EDIT" if label == "GENUINE" else "TAMPER",
                    "identifier": identity["identifier"],
                    "specimen_label": "SPECIMEN – NOT A VALID DOCUMENT",
                }
            )

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    expected_counts = {
        "images": 8,
        "identities": 4,
        "pairs": 4,
        "label_counts": {"GENUINE": 4, "TAMPERED": 4},
        "category_counts": {"TEXT": 8, "PHOTO": 0},
        "template_counts": {
            template: sum(row["template_id"] == template for row in rows)
            for template in sorted({row["template_id"] for row in rows})
        },
        "split_counts": {"train": 4, "test": 4},
    }
    config = AuditConfig(
        manifest_path=manifest_path,
        data_root=data_root,
        splits_path=splits_path,
        report_path=report_path,
        expected_counts=expected_counts,
        image_size=IMAGE_SIZE,
    )
    return config, rows, manifest_path


def _write_manifest(manifest_path: Path, rows: list[dict]) -> None:
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_passes_for_consistent_mini_dataset(tmp_path):
    config, _, _ = _write_mini_dataset(tmp_path)

    report = run_audit(config)

    assert report["status"] == "PASS"
    assert report["summary"]["image_count"] == 8
    assert config.report_path.exists()
    assert json.loads(config.report_path.read_text(encoding="utf-8")) == report


def test_audit_detects_missing_image(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    (config.data_root / rows[0]["image_path"]).unlink()
    _write_manifest(manifest_path, rows)

    with pytest.raises(AuditError, match="file_integrity"):
        run_audit(config)


def test_audit_detects_class_imbalance(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    rows[0]["label"] = "TAMPERED"
    _write_manifest(manifest_path, rows)

    with pytest.raises(AuditError, match="balance"):
        run_audit(config)


def test_audit_detects_pair_degradation_mismatch(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    rows[1]["category"] = "PHOTO"
    _write_manifest(manifest_path, rows)

    with pytest.raises(AuditError, match="pair_integrity"):
        run_audit(config)


def test_audit_detects_split_source_mismatch(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    rows[-1]["split"] = "validation"
    _write_manifest(manifest_path, rows)

    with pytest.raises(AuditError, match="leakage"):
        run_audit(config)


def test_audit_detects_duplicate_images(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    source = config.data_root / rows[0]["image_path"]
    target = config.data_root / rows[2]["image_path"]
    target.write_bytes(source.read_bytes())

    with pytest.raises(AuditError, match="duplicates"):
        run_audit(config)


def test_audit_detects_invalid_identifier(tmp_path):
    config, rows, manifest_path = _write_mini_dataset(tmp_path)
    rows[1]["template_family"] = "PAN"
    rows[1]["identifier"] = "ABCPD1234F"
    _write_manifest(manifest_path, rows)

    with pytest.raises(AuditError, match="identifier_safety"):
        run_audit(config)
