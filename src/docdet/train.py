"""CPU-first Keras training loop for the compact CNN."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from docdet.config import resolve_project_path
from docdet.dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MANIFEST_PATH,
    SyntheticDataset,
    to_tf_dataset,
)
from docdet.evaluate import collect_predictions, compute_metrics
from docdet.model import DEFAULT_L2, build_model, set_determinism

DEFAULT_CHECKPOINT_PATH = "results/checkpoints/best.weights.h5"
DEFAULT_HISTORY_PATH = "results/training_history.json"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 25
    learning_rate: float = 3e-4
    l2: float = DEFAULT_L2
    early_stopping_patience: int | None = None
    checkpoint_path: Path = Path(DEFAULT_CHECKPOINT_PATH)
    history_path: Path = Path(DEFAULT_HISTORY_PATH)
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive or None")
        object.__setattr__(
            self, "checkpoint_path", resolve_project_path(self.checkpoint_path)
        )
        object.__setattr__(
            self, "history_path", resolve_project_path(self.history_path)
        )


@dataclass(frozen=True)
class TrainingResult:
    model: tf.keras.Model
    history: dict[str, list[float]]
    checkpoint_path: Path
    duration_seconds: float
    best_val_loss: float


class _MetricsLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        train_data: SyntheticDataset | None,
        validation_data: SyntheticDataset | None,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _collect(self, data: SyntheticDataset | None):
        if data is None:
            return None
        cached = self._cache.get(id(data))
        if cached is None:
            images: list[np.ndarray] = []
            labels: list[int] = []
            for sample in data:
                images.append(sample.image)
                labels.append(int(sample.label))
            if not images:
                return None
            cached = (
                np.stack(images).astype(np.float32),
                np.asarray(labels, dtype=np.int64),
            )
            self._cache[id(data)] = cached
        stacked, labels = cached
        probs = np.asarray(self.model.predict(stacked, verbose=0)).reshape(-1)
        return probs, labels

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        logs = logs if logs is not None else {}
        for name, data in (("train", self.train_data), ("validation", self.validation_data)):
            collected = self._collect(data)
            if collected is None:
                continue
            probs, labels = collected
            metrics = compute_metrics(labels, probs, 0.5)
            logs[f"{name}_accuracy"] = float(metrics["accuracy"])
            logs[f"{name}_auc"] = float(metrics["roc_auc"])
            logs[f"{name}_score_mean"] = float(probs.mean())
            logs[f"{name}_score_std"] = float(probs.std())
        logs["learning_rate"] = float(
            tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        )


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    config: TrainingConfig,
    *,
    train_metrics_data: SyntheticDataset | None = None,
    validation_metrics_data: SyntheticDataset | None = None,
) -> TrainingResult:
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.history_path.parent.mkdir(parents=True, exist_ok=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]
    if config.early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                mode="min",
                restore_best_weights=True,
                verbose=0,
            )
        )
    if train_metrics_data is not None or validation_metrics_data is not None:
        callbacks.append(
            _MetricsLoggerCallback(
                train_metrics_data,
                validation_metrics_data,
                config.batch_size,
            )
        )

    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.max_epochs,
        steps_per_epoch=config.steps_per_epoch,
        validation_steps=config.validation_steps,
        callbacks=callbacks,
        verbose=config.verbose,
    )
    duration = time.perf_counter() - started

    serialized_history = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }
    best_val_loss = min(serialized_history["val_loss"])
    config.history_path.write_text(
        json.dumps(
            {
                **serialized_history,
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(config).items()
                },
                "duration_seconds": duration,
                "best_val_loss": best_val_loss,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return TrainingResult(
        model=model,
        history=serialized_history,
        checkpoint_path=config.checkpoint_path,
        duration_seconds=duration,
        best_val_loss=best_val_loss,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CNN on CPU")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--history", default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    args = parser.parse_args(argv)

    set_determinism(args.seed)
    train_data = SyntheticDataset(args.manifest, args.data_root, split="train")
    validation_data = SyntheticDataset(args.manifest, args.data_root, split="validation")
    train_dataset = to_tf_dataset(train_data, batch_size=args.batch_size, repeat=True)
    train_dataset = train_dataset.shuffle(buffer_size=max(1, len(train_data)))
    validation_dataset = to_tf_dataset(validation_data, batch_size=args.batch_size)
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = ceil(len(train_data) / args.batch_size)
    validation_steps = args.validation_steps
    if validation_steps is None:
        validation_steps = ceil(len(validation_data) / args.batch_size)
    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=None,
        checkpoint_path=Path(args.checkpoint),
        history_path=Path(args.history),
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )
    result = train_model(
        build_model(l2=config.l2),
        train_dataset,
        validation_dataset,
        config,
        train_metrics_data=train_data,
        validation_metrics_data=validation_data,
    )
    print(
        json.dumps(
            {
                "epochs_run": len(result.history["loss"]),
                "best_val_loss": result.best_val_loss,
                "duration_seconds": result.duration_seconds,
                "checkpoint": str(result.checkpoint_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_HISTORY_PATH",
    "TrainingConfig",
    "TrainingResult",
    "train_model",
]
