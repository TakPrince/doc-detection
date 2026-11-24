"""Model inference and specimen-selection helpers for the ID-Guard UI."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from docdet.config import resolve_project_path
from docdet.preprocessing import preprocess_image

DEFAULT_CHECKPOINT_PATH = "results/phase5_3/checkpoints/best.weights.h5"
DEFAULT_THRESHOLD_ARTIFACT_PATH = "results/phase5_3/evaluation.json"
DEFAULT_MANIFEST_PATH = "data/manifests/dataset.jsonl"
DEFAULT_DATA_ROOT = "data/generated"
DEFAULT_THRESHOLD = 0.8840

TAMPERED_LABEL = "Potentially tampered"
UNALTERED_LABEL = "Classified as content-unaltered"
DOCUMENT_TYPE_FAMILIES = {"Aadhaar": "AAD", "PAN": "PAN"}
DOCUMENT_TYPES = tuple(DOCUMENT_TYPE_FAMILIES)


class InferenceError(RuntimeError):
    """Raised when a specimen cannot be analyzed safely."""


class ModelUnavailableError(InferenceError):
    """Raised when the trained model cannot be loaded."""


@dataclass(frozen=True)
class Specimen:
    image_id: str
    image_path: Path
    template_family: str
    template_id: str
    category: str
    display_name: str


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    threshold: float


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def apply_threshold(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Return whether a model score is classified as potentially tampered."""
    normalized_score = _probability(score, "score")
    normalized_threshold = _probability(threshold, "threshold")
    return normalized_score >= normalized_threshold


def prediction_label(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Map a model score to the safe user-facing classification label."""
    if apply_threshold(score, threshold):
        return TAMPERED_LABEL
    return UNALTERED_LABEL


def load_threshold(
    artifact_path: str | Path = DEFAULT_THRESHOLD_ARTIFACT_PATH,
) -> float:
    """Load the saved validation threshold, falling back to the frozen value."""
    path = resolve_project_path(artifact_path)
    if not path.exists():
        return DEFAULT_THRESHOLD
    if not path.is_file():
        raise ValueError("The saved threshold artifact is not a file.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("The saved threshold artifact could not be read.") from exc
    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError("The saved threshold artifact does not contain a threshold.")
    return _probability(payload["threshold"], "threshold")


def _safe_image_path(data_root: Path, image_path: str) -> Path:
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The specimen manifest contains an unsafe image path.")
    root = data_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("The specimen manifest contains an unsafe image path.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Specimen image is missing: {image_path}")
    return candidate


def load_specimens(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    document_type: str = "Aadhaar",
) -> list[Specimen]:
    """Load the generated specimens matching a selected document type."""
    family = DOCUMENT_TYPE_FAMILIES.get(document_type)
    if family is None:
        raise ValueError(f"Unsupported document type: {document_type}")
    manifest = resolve_project_path(manifest_path)
    root = resolve_project_path(data_root)
    if not manifest.is_file():
        raise FileNotFoundError(f"Specimen manifest is missing: {manifest}")
    specimens: list[Specimen] = []
    seen_ids: set[str] = set()
    pair_variants: dict[str, int] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"Specimen manifest could not be read: {manifest}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Specimen manifest is invalid at line {line_number}.") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"Specimen manifest is invalid at line {line_number}.")
        if entry.get("template_family") != family:
            continue
        try:
            image_id = str(entry["image_id"])
            image_path = str(entry["image_path"])
            template_id = str(entry["template_id"])
            category = str(entry["category"])
        except KeyError as exc:
            raise ValueError(f"Specimen manifest is incomplete at line {line_number}.") from exc
        if image_id in seen_ids:
            raise ValueError(f"Specimen manifest contains duplicate ID: {image_id}")
        seen_ids.add(image_id)
        identity_id = str(entry.get("identity_id", image_id))
        pair_id = str(entry.get("pair_id", image_id))
        variant = pair_variants.get(pair_id, 0) + 1
        pair_variants[pair_id] = variant
        specimens.append(
            Specimen(
                image_id=image_id,
                image_path=_safe_image_path(root, image_path),
                template_family=family,
                template_id=template_id,
                category=category,
                display_name=(
                    f"{identity_id} · {category.title()} · Variant {variant}"
                ),
            )
        )
    if not specimens:
        raise ValueError(f"No generated specimens are available for {document_type}.")
    return specimens


def select_specimen(specimens: Sequence[Specimen], image_id: str) -> Specimen:
    """Return the selected specimen by its stable manifest ID."""
    for specimen in specimens:
        if specimen.image_id == image_id:
            return specimen
    raise KeyError("The selected specimen is not available.")


def load_trained_model(checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH):
    """Build the existing CNN and load the Phase 5.3 weights."""
    checkpoint = resolve_project_path(checkpoint_path)
    if not checkpoint.is_file():
        raise ModelUnavailableError(
            f"The trained model checkpoint is missing: {checkpoint}"
        )
    try:
        from docdet.model import build_model

        model = build_model()
        model.load_weights(str(checkpoint))
    except ImportError as exc:
        raise ModelUnavailableError(
            "TensorFlow is unavailable, so the trained model cannot be loaded."
        ) from exc
    except Exception as exc:
        raise ModelUnavailableError(
            "The trained model checkpoint could not be loaded."
        ) from exc
    return model


def predict_image(
    model: Any,
    image: Image.Image | str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> Prediction:
    """Preprocess and classify a generated or uploaded image."""
    try:
        processed = preprocess_image(image)
        probabilities = model.predict(processed[np.newaxis, ...], verbose=0)
        score = float(np.asarray(probabilities).reshape(-1)[0])
    except Exception as exc:
        raise InferenceError(
            "The model could not analyze the uploaded image."
        ) from exc
    normalized_score = _probability(score, "model score")
    normalized_threshold = _probability(threshold, "threshold")
    return Prediction(
        label=prediction_label(normalized_score, normalized_threshold),
        score=normalized_score,
        threshold=normalized_threshold,
    )


def predict_specimen(
    model: Any,
    image_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> Prediction:
    """Preprocess and classify one generated specimen."""
    return predict_image(model, image_path, threshold)


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLD_ARTIFACT_PATH",
    "DOCUMENT_TYPES",
    "DOCUMENT_TYPE_FAMILIES",
    "InferenceError",
    "ModelUnavailableError",
    "Prediction",
    "Specimen",
    "TAMPERED_LABEL",
    "UNALTERED_LABEL",
    "apply_threshold",
    "load_specimens",
    "load_threshold",
    "load_trained_model",
    "predict_image",
    "predict_specimen",
    "prediction_label",
    "select_specimen",
]
