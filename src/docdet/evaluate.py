"""Basic test-set evaluation for the tamper-detection CNN."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from docdet.config import resolve_project_path
from docdet.dataset import DEFAULT_DATA_ROOT, DEFAULT_MANIFEST_PATH, SyntheticDataset
from docdet.model import set_determinism

DEFAULT_RESULTS_PATH = "results/evaluation.json"
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class EvaluationConfig:
    results_path: Path = Path(DEFAULT_RESULTS_PATH)
    batch_size: int = 32
    threshold: float = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        object.__setattr__(
            self, "results_path", resolve_project_path(self.results_path)
        )


@dataclass(frozen=True)
class PredictionBatch:
    probabilities: np.ndarray
    labels: np.ndarray
    metadata: list[dict[str, Any]]


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob), dtype=np.float64)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and y_prob[order[end]] == y_prob[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def _pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-y_prob)
    ordered_true = y_true[order]
    true_positives = np.cumsum(ordered_true)
    precision = true_positives / np.arange(1, len(ordered_true) + 1)
    return float(np.sum(precision[ordered_true == 1]) / positives)


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if truth.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    predicted = (probabilities >= threshold).astype(np.int64)

    true_negative = int(np.sum((truth == 0) & (predicted == 0)))
    false_positive = int(np.sum((truth == 0) & (predicted == 1)))
    false_negative = int(np.sum((truth == 1) & (predicted == 0)))
    true_positive = int(np.sum((truth == 1) & (predicted == 1)))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )

    return {
        "accuracy": float(np.mean(predicted == truth)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(_roc_auc(truth, probabilities)),
        "pr_auc": float(_pr_auc(truth, probabilities)),
        "specificity": float(specificity),
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def collect_predictions(
    model: Any,
    dataset: Iterable[Any],
    batch_size: int = 32,
) -> PredictionBatch:
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    images: list[np.ndarray] = []

    def flush() -> None:
        if not images:
            return
        values = np.asarray(model.predict(np.stack(images), verbose=0)).reshape(-1)
        probabilities.append(values.astype(np.float64))

    for sample in dataset:
        images.append(sample.image)
        labels.append(int(sample.label))
        metadata.append(dict(sample.metadata))
        if len(images) == batch_size:
            flush()
            images.clear()
    flush()

    if not labels:
        raise ValueError("dataset is empty")
    return PredictionBatch(
        probabilities=np.concatenate(probabilities),
        labels=np.asarray(labels, dtype=np.int64),
        metadata=metadata,
    )


def select_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> float:
    truth = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0
    for candidate in np.unique(probabilities):
        metrics = compute_metrics(truth, probabilities, float(candidate))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(candidate)
    return best_threshold


def _null_edit_fpr(batch: PredictionBatch, threshold: float) -> float:
    genuine = [
        index
        for index, item in enumerate(batch.metadata)
        if item.get("operation") == "NULL_EDIT"
    ]
    if not genuine:
        return 0.0
    return float(np.mean(batch.probabilities[genuine] >= threshold))


def evaluate_pipeline(
    model: Any,
    validation_dataset: Iterable[Any],
    test_dataset: Iterable[Any],
    config: EvaluationConfig,
) -> dict[str, Any]:
    validation = collect_predictions(model, validation_dataset, config.batch_size)
    threshold = select_threshold(validation.labels, validation.probabilities)

    test = collect_predictions(model, test_dataset, config.batch_size)
    report = {
        "schema_version": 1,
        "threshold": threshold,
        "threshold_source": "validation",
        "threshold_rule": "maximum_f1",
        "validation": {
            "metrics": compute_metrics(
                validation.labels, validation.probabilities, threshold
            ),
        },
        "test": {
            "metrics": compute_metrics(test.labels, test.probabilities, threshold),
            "null_edit_fpr": _null_edit_fpr(test, threshold),
        },
    }
    config.results_path.parent.mkdir(parents=True, exist_ok=True)
    config.results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN")
    parser.add_argument("--checkpoint", default="results/checkpoints/best.weights.h5")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from docdet.model import build_model

    set_determinism(args.seed)
    model = build_model()
    model.load_weights(resolve_project_path(args.checkpoint))
    config = EvaluationConfig(
        results_path=Path(args.results),
        batch_size=args.batch_size,
    )
    report = evaluate_pipeline(
        model,
        SyntheticDataset(args.manifest, args.data_root, split="validation"),
        SyntheticDataset(args.manifest, args.data_root, split="test"),
        config,
    )
    print(json.dumps(report["test"]["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RESULTS_PATH",
    "DEFAULT_THRESHOLD",
    "EvaluationConfig",
    "PredictionBatch",
    "collect_predictions",
    "compute_metrics",
    "evaluate_pipeline",
    "select_threshold",
]
