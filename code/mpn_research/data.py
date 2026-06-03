"""Dataset loading, stratified splits, and tf.data pipelines with augmentation."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

from . import config


def _collect_image_paths(root: Path):
    """Return parallel lists of image paths and integer labels (0..NUM_CLASSES-1)."""
    paths: list[str] = []
    labels: list[int] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for idx, name in enumerate(config.CLASS_NAMES):
        class_dir = root / name
        if not class_dir.is_dir():
            continue
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in exts:
                paths.append(str(p))
                labels.append(idx)
    if not paths:
        raise FileNotFoundError(
            f"No images found under {root} with subfolders {config.CLASS_NAMES}."
        )
    return paths, labels


def _decode_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, config.IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def _make_augment_fn():
    """
    Strong on-the-fly augmentation for small histopathology datasets.
    Applied only on the training split (training=True). Val/test stay unaugmented.
    Pixel range is ~[0, 255] float32 before backbone-specific preprocess_input.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            # factor is fraction of 2π — ~0.35 ≈ ±63° max rotation
            tf.keras.layers.RandomRotation(0.35),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.35, 0.35),
                width_factor=(-0.35, 0.35),
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomContrast(0.35),
            tf.keras.layers.RandomBrightness(0.35, value_range=(0.0, 255.0)),
        ],
        name="heavy_augmentation",
    )


def get_preprocess_fn():
    """Return preprocessing aligned with the chosen ImageNet backbone."""
    name = config.BACKBONE
    if name == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input
    if name == "DenseNet121":
        return tf.keras.applications.densenet.preprocess_input
    if name == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    raise ValueError(f"Unknown BACKBONE: {name}")


def prepare_datasets(
    batch_size: int,
    seed: int | None = None,
):
    """
    Stratified split:
    - 80% (train+val pool) vs 20% test
    - From the 80% pool: 80% train vs 20% validation
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    paths, labels = _collect_image_paths(config.DATASET_DIR)
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=config.TEST_SIZE,
        stratify=labels,
        random_state=seed,
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=config.VAL_SIZE_WITHIN_TRAIN,
        stratify=train_val_labels,
        random_state=seed,
    )

    preprocess = get_preprocess_fn()
    augment = _make_augment_fn()

    def _apply_preprocess(image, label):
        return preprocess(image), label

    def _train_map(path, label):
        image, label = _decode_resize(path, label)
        image = augment(image, training=True)
        return _apply_preprocess(image, label)

    def _eval_map(path, label):
        image, label = _decode_resize(path, label)
        return _apply_preprocess(image, label)

    autotune = tf.data.AUTOTUNE
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(len(train_paths), seed=seed, reshuffle_each_iteration=True)
        .map(_train_map, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_eval_map, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
        .map(_eval_map, num_parallel_calls=autotune)
        .batch(batch_size)
        .prefetch(autotune)
    )

    meta = {
        "n_train": int(len(train_paths)),
        "n_val": int(len(val_paths)),
        "n_test": int(len(test_paths)),
        "train_paths": train_paths,
        "test_paths": test_paths,
        "test_labels": test_labels,
    }
    return train_ds, val_ds, test_ds, meta
