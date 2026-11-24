"""Compact CNN for synthetic document tamper detection.

Layer patterns follow the official Keras API:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Model
"""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf

MODEL_INPUT_SHAPE = (160, 256, 3)
DEFAULT_L2 = 1e-4
CONV_FILTERS = (16, 32, 64, 128)


def set_determinism(seed: int = 42) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def build_model(
    input_shape: tuple[int, int, int] = MODEL_INPUT_SHAPE,
    l2: float = DEFAULT_L2,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(l2)
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(scale=1.0, name="rescaling")(inputs)

    for index, filters in enumerate(CONV_FILTERS, start=1):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            kernel_regularizer=regularizer,
            name=f"conv_{index}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{index}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{index}")(x)
        if index < len(CONV_FILTERS):
            x = tf.keras.layers.MaxPooling2D(pool_size=2, name=f"pool_{index}")(x)

    average = tf.keras.layers.GlobalAveragePooling2D(name="global_average")(x)
    maximum = tf.keras.layers.GlobalMaxPooling2D(name="global_max")(x)
    x = tf.keras.layers.Concatenate(name="pool_concat")([average, maximum])
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizer,
        name="dense_64",
    )(x)
    x = tf.keras.layers.Dropout(rate=0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="id_guard_cnn")


__all__ = ["CONV_FILTERS", "DEFAULT_L2", "MODEL_INPUT_SHAPE", "build_model", "set_determinism"]
