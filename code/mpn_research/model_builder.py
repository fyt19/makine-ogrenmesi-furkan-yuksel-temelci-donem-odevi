"""Transfer-learning CNN builder and compile helpers."""

from __future__ import annotations

import tensorflow as tf

from . import config


def _base_model(input_shape):
    name = config.BACKBONE
    kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        name="backbone",
    )
    if name == "ResNet50":
        return tf.keras.applications.ResNet50(**kwargs)
    if name == "DenseNet121":
        return tf.keras.applications.DenseNet121(**kwargs)
    if name == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(**kwargs)
    raise ValueError(f"Unknown BACKBONE: {name}")


def build_model(learning_rate: float, dropout_rate: float) -> tf.keras.Model:
    """
    Baseline-style CNN: frozen ImageNet backbone + small classification head.
    """
    tf.keras.backend.clear_session()
    base = _base_model((*config.IMG_SIZE, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="fc_hidden")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=f"mpn_{config.BACKBONE.lower()}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
