"""Grad-CAM visual explanations for pathology CNNs (TensorFlow 2 / Keras)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

from . import config
from .data import get_preprocess_fn


def _inner_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """
    Return the frozen Applications backbone wrapped in the classifier.

    Prefer ``get_layer("backbone")`` (matches ``model_builder``). If names differ,
    fall back to the first nested ``Model`` among ``model.layers`` (e.g. index 0/1).
    """
    try:
        maybe = model.get_layer("backbone")
        if isinstance(maybe, tf.keras.Model):
            return maybe
    except ValueError:
        pass
    for lay in model.layers:
        if isinstance(lay, tf.keras.Model):
            return lay
    raise ValueError("Could not resolve an inner backbone Model on the classifier.")


def _unwrap_layer_input(inp):
    if isinstance(inp, (list, tuple)):
        if len(inp) != 1:
            raise ValueError("Expected a single tensor into GAP for Grad-CAM.")
        return inp[0]
    return inp


def _spatial_tensor_connected_to_inputs(model: tf.keras.Model):
    """
    Tensor used for Grad-CAM must sit on the same graph as ``model.input``.

    With a nested ``base(inputs)`` Applications model, ``backbone.output`` or
    ``backbone.get_layer(...).output`` often refers to the backbone's *standalone*
    subgraph (Keras 3: ``ValueError: Output ... is not connected to inputs``).

    Strategy:
    1. ResNet50: if ``conv5_block3_out`` is connectable to ``model.input``, use it
       (classic last spatial activation before pooling).
    2. Otherwise use ``gap.input`` — the spatial map feeding ``GlobalAveragePooling2D``
       on the **outer** functional graph (equivalent to backbone output before GAP).
    """
    gap = model.get_layer("gap")
    gap_spatial = _unwrap_layer_input(gap.input)

    if config.BACKBONE != "ResNet50":
        return gap_spatial

    backbone = _inner_backbone(model)
    if not any(lyr.name == "conv5_block3_out" for lyr in backbone.layers):
        return gap_spatial

    conv_out = backbone.get_layer("conv5_block3_out").output
    try:
        tf.keras.Model(
            inputs=model.input,
            outputs=[conv_out, model.output],
            name="_gradcam_connectivity_probe",
        )
        return conv_out
    except ValueError:
        return gap_spatial


def _ensure_4d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        return np.expand_dims(x, axis=0)
    return x


def compute_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    Grad-CAM heatmap (H, W) from gradients of the target class w.r.t. a spatial
    feature map that is **reachable from** ``model.input`` (Keras 3-safe).

    ``img_batch`` must already match training preprocessing (e.g. ResNet preprocess_input).
    """
    spatial = _spatial_tensor_connected_to_inputs(model)
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[spatial, model.output],
        name="gradcam_submodel",
    )

    img_batch = tf.cast(_ensure_4d(img_batch), tf.float32)
    img_batch = img_batch[:1]

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_batch, training=False)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]).numpy())
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Gradients w.r.t. the Grad-CAM feature map are None; check model connectivity."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    heatmap = tf.squeeze(heatmap, axis=0)
    heatmap = tf.nn.relu(heatmap)
    max_h = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_h + 1e-8)
    return heatmap.numpy()


def overlay_gradcam_on_image(
    display_image_01: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Superimpose jet heatmap on RGB image in [0, 1]. Returns float32 RGB [0,1]."""
    h, w = display_image_01.shape[0], display_image_01.shape[1]
    heat_small = tf.image.resize(
        np.expand_dims(heatmap, -1), (h, w), method="bilinear"
    )
    heat_small = tf.squeeze(heat_small, axis=-1).numpy()
    heat_u8 = np.clip(np.uint8(255 * heat_small), 0, 255)
    jet = plt.get_cmap("jet")(heat_u8 / 255.0)[..., :3].astype(np.float32)
    out = jet * alpha + display_image_01.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0.0, 1.0)


def save_gradcam_for_path(
    image_path: str,
    model: tf.keras.Model,
    out_path: Path,
    true_class_name: str | None = None,
) -> Path:
    """
    Load an RGB image from disk, preprocess, run Grad-CAM for the predicted class,
    and save an input | jet-overlay figure.
    """
    img = load_img(image_path, target_size=config.IMG_SIZE)
    img_arr = img_to_array(img).astype(np.float32)
    display_01 = img_arr / 255.0
    preprocess = get_preprocess_fn()
    batch = np.expand_dims(preprocess(img_arr.copy()), axis=0)

    preds = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    heatmap = compute_gradcam(model, batch, pred_index=pred_idx)

    overlaid = overlay_gradcam_on_image(display_01, heatmap, alpha=0.45)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(display_01)
    axes[0].set_title("Input (RGB)")
    axes[0].axis("off")

    axes[1].imshow(overlaid)
    title = f"Grad-CAM (jet) — pred: {config.CLASS_NAMES[pred_idx]}"
    if true_class_name:
        title += f" | true: {true_class_name}"
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_gradcam_per_class_grid(
    model: tf.keras.Model,
    sample_rows: list[tuple[str, int]],
    out_path: Path,
    alpha: float = 0.45,
) -> Path:
    """
    One figure: ``len(sample_rows)`` rows × (1 + NUM_CLASSES) columns.

    Each row is one test image: RGB input, then Grad-CAM (jet overlay) targeting
    each class score (PV, ET, MF) so class-specific saliency can be compared.

    ``sample_rows`` items are ``(image_path, true_class_index)``.
    """
    n = len(sample_rows)
    if n == 0:
        raise ValueError("sample_rows must be non-empty.")
    n_cls = config.NUM_CLASSES
    fig_w = 3.6 * (1 + n_cls)
    fig_h = 2.9 * n
    fig, axes = plt.subplots(n, 1 + n_cls, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.reshape(axes, (1, -1))

    col_labels = ["Input (RGB)"] + [
        f"Grad-CAM → {config.CLASS_NAMES[c]}" for c in range(n_cls)
    ]
    for j, lab in enumerate(col_labels):
        axes[0, j].set_title(lab, fontsize=10)

    preprocess = get_preprocess_fn()
    for i, (path, true_y) in enumerate(sample_rows):
        img = load_img(path, target_size=config.IMG_SIZE)
        img_arr = img_to_array(img).astype(np.float32)
        display_01 = img_arr / 255.0
        batch = np.expand_dims(preprocess(img_arr.copy()), axis=0)
        preds = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        axes[i, 0].imshow(display_01)
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(
            f"#{i + 1}\ntrue: {config.CLASS_NAMES[true_y]}\n"
            f"pred: {config.CLASS_NAMES[pred_idx]}",
            fontsize=8,
            rotation=0,
            labelpad=8,
            va="center",
        )

        for ci in range(n_cls):
            heat = compute_gradcam(model, batch, pred_index=ci)
            overlaid = overlay_gradcam_on_image(display_01, heat, alpha=alpha)
            axes[i, 1 + ci].imshow(overlaid)
            axes[i, 1 + ci].axis("off")
            axes[i, 1 + ci].set_xlabel(f"p({config.CLASS_NAMES[ci]})={preds[ci]:.2f}", fontsize=8)

    fig.suptitle(
        f"Grad-CAM (jet) — {n} test inputs × {n_cls} class targets — backbone {config.BACKBONE}",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
