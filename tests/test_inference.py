from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from docdet.inference import (
    DEFAULT_THRESHOLD,
    TAMPERED_LABEL,
    UNALTERED_LABEL,
    InferenceError,
    ModelUnavailableError,
    apply_threshold,
    load_specimens,
    load_threshold,
    load_trained_model,
    predict_image,
    predict_specimen,
    prediction_label,
    select_specimen,
)
from docdet.upload import (
    MAX_UPLOAD_BYTES,
    UploadValidationError,
    decode_uploaded_image,
)


def _write_specimen(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-image")


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_apply_threshold_uses_inclusive_boundary():
    assert apply_threshold(0.8839, threshold=0.8840) is False
    assert apply_threshold(0.8840, threshold=0.8840) is True


def test_prediction_labels_match_threshold_decision():
    assert prediction_label(0.8840, threshold=0.8840) == TAMPERED_LABEL
    assert prediction_label(0.8839, threshold=0.8840) == UNALTERED_LABEL


def test_load_threshold_prefers_saved_artifact_and_uses_frozen_fallback(tmp_path):
    artifact = tmp_path / "evaluation.json"
    artifact.write_text(json.dumps({"threshold": 0.71}), encoding="utf-8")

    assert load_threshold(artifact) == pytest.approx(0.71)
    assert load_threshold(tmp_path / "missing.json") == pytest.approx(DEFAULT_THRESHOLD)


def test_specimen_selection_filters_document_type(tmp_path):
    data_root = tmp_path / "generated"
    manifest_path = tmp_path / "manifest.jsonl"
    aadhaar_path = data_root / "images" / "aadhaar.png"
    pan_path = data_root / "images" / "pan.png"
    _write_specimen(aadhaar_path)
    _write_specimen(pan_path)
    _write_manifest(
        manifest_path,
        [
            {
                "image_id": "aadhaar-1",
                "identity_id": "AAD-1-0001",
                "pair_id": "AAD-1-0001__TEXT",
                "image_path": "images/aadhaar.png",
                "template_family": "AAD",
                "template_id": "AAD-1",
                "category": "TEXT",
            },
            {
                "image_id": "pan-1",
                "identity_id": "PAN-1-0001",
                "pair_id": "PAN-1-0001__PHOTO",
                "image_path": "images/pan.png",
                "template_family": "PAN",
                "template_id": "PAN-1",
                "category": "PHOTO",
            },
        ],
    )

    specimens = load_specimens(manifest_path, data_root, document_type="Aadhaar")

    assert [specimen.image_id for specimen in specimens] == ["aadhaar-1"]
    selected = select_specimen(specimens, "aadhaar-1")
    assert selected.image_path == aadhaar_path
    assert selected.display_name == "AAD-1-0001 · Text · Variant 1"


def test_specimen_selection_rejects_unknown_document_type(tmp_path):
    with pytest.raises(ValueError, match="document type"):
        load_specimens(
            tmp_path / "manifest.jsonl",
            tmp_path / "generated",
            document_type="Passport",
        )


def test_missing_model_checkpoint_has_clear_error(tmp_path):
    with pytest.raises(ModelUnavailableError, match="checkpoint is missing"):
        load_trained_model(tmp_path / "missing.weights.h5")


def test_inference_failure_has_clear_error(tmp_path):
    image_path = tmp_path / "specimen.png"
    Image.new("RGB", (100, 60), (20, 40, 60)).save(image_path)

    class FailingModel:
        def predict(self, images, verbose=0):
            raise RuntimeError("model failed")

    with pytest.raises(InferenceError, match="could not analyze"):
        predict_specimen(FailingModel(), image_path, threshold=DEFAULT_THRESHOLD)


def test_uploaded_png_is_decoded_as_rgb():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), (20, 40, 60)).save(buffer, format="PNG")

    image = decode_uploaded_image(buffer.getvalue())

    assert image.mode == "RGB"
    assert image.size == (100, 60)


def test_uploaded_image_rejects_invalid_content():
    with pytest.raises(UploadValidationError, match="valid PNG or JPEG"):
        decode_uploaded_image(b"not-an-image")


def test_uploaded_image_rejects_disallowed_format():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), (20, 40, 60)).save(buffer, format="GIF")

    with pytest.raises(UploadValidationError, match="valid PNG or JPEG"):
        decode_uploaded_image(buffer.getvalue())


def test_uploaded_image_rejects_oversized_content():
    with pytest.raises(UploadValidationError, match="10 MB"):
        decode_uploaded_image(b"x" * (MAX_UPLOAD_BYTES + 1))


def test_predict_image_uses_existing_preprocessing():
    class RedChannelModel:
        def predict(self, images, verbose=0):
            return [[float(images[0, 0, 0, 0])]]

    image = Image.new("RGB", (100, 60), (255, 0, 0))

    prediction = predict_image(RedChannelModel(), image, threshold=DEFAULT_THRESHOLD)

    assert prediction.label == TAMPERED_LABEL
    assert prediction.score == pytest.approx(1.0)
