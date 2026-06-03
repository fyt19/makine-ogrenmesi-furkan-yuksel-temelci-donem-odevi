"""
Ph-negative MPN histopathology classification:
Baseline CNN vs. CNN+PSO vs. CNN+GWO vs. CNN+Hybrid PSO-GWO, with Grad-CAM XAI.

Run from the ``code/`` directory:

    cd code
    python main.py

Regenerate Grad-CAM only (loads saved ``weights_<slug>.weights.h5``, no training):

    python main.py --gradcam-only --weights-slug baseline

For checkpoints trained with PSO/GWO/Hybrid hyperparameters, pass the same
``--learning-rate`` and ``--dropout`` you used in that run so layer shapes match:

    python main.py --gradcam-only --weights-slug pso --learning-rate 0.0038 --dropout 0.44

Custom weights file:

    python main.py --gradcam-only --weights-path /path/to/weights_baseline.weights.h5 --dropout 0.5

Five random test images, each with input + Grad-CAM toward PV / ET / MF (one PNG grid):

    python main.py --gradcam-grid --weights-slug baseline
"""

from __future__ import annotations

import argparse
import random
import ssl
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

ssl._create_default_https_context = ssl._create_unverified_context

from mpn_research import config
from mpn_research import data as data_mod
from mpn_research import gradcam as gradcam_mod
from mpn_research import meta_heuristics
from mpn_research import model_builder
from mpn_research import paper_outputs


def set_global_seeds(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _hp_from_dict(lr: float, dropout: float, batch_size: int) -> dict:
    return {
        "learning_rate": lr,
        "dropout": dropout,
        "batch_size": batch_size,
    }


def _weights_path(slug: str) -> Path:
    return config.OUTPUT_DIR / f"weights_{slug}.weights.h5"


def train_evaluate_full(
    name: str,
    learning_rate: float,
    dropout: float,
    batch_size: int,
    epochs: int,
    weights_slug: str | None = None,
    verbose_fit: int = 1,
) -> tuple[dict, np.ndarray, np.ndarray, Path | None]:
    """
    Train on train split, monitor val, evaluate on held-out test.
    Returns (history_dict, y_true, y_proba, weights_path_or_none).
    """
    _ = name  # kept for readable call sites / future logging hooks
    train_ds, val_ds, test_ds, meta = data_mod.prepare_datasets(
        batch_size=batch_size, seed=config.RANDOM_SEED
    )
    y_true = np.asarray(meta["test_labels"], dtype=np.int32)

    model = model_builder.build_model(
        learning_rate=learning_rate,
        dropout_rate=dropout,
    )

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=verbose_fit,
    )

    probs = model.predict(test_ds, verbose=0)

    wpath: Path | None = None
    if weights_slug is not None:
        wpath = _weights_path(weights_slug)
        model.save_weights(str(wpath))

    h = hist.history
    if "accuracy" in h:
        acc = h["accuracy"]
    elif "sparse_categorical_accuracy" in h:
        acc = h["sparse_categorical_accuracy"]
    else:
        raise KeyError("No accuracy key found in Keras history.")

    if "val_accuracy" in h:
        val_acc = h["val_accuracy"]
    elif "val_sparse_categorical_accuracy" in h:
        val_acc = h["val_sparse_categorical_accuracy"]
    else:
        raise KeyError("No val_accuracy key found in Keras history.")

    history_serializable = {
        "loss": list(map(float, h["loss"])),
        "val_loss": list(map(float, h["val_loss"])),
        "accuracy": list(map(float, acc)),
        "val_accuracy": list(map(float, val_acc)),
    }

    del model
    tf.keras.backend.clear_session()

    return history_serializable, y_true, probs, wpath


def _pick_n_random_test_samples(meta: dict, n: int, seed: int) -> list[tuple[str, int]]:
    """``n`` distinct test paths with their integer labels (for Grad-CAM grids)."""
    paths = np.asarray(meta["test_paths"])
    labels = np.asarray(meta["test_labels"], dtype=np.int32)
    if n > len(paths):
        raise ValueError(f"Requested {n} test samples but only {len(paths)} exist.")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), size=n, replace=False)
    return [(str(paths[i]), int(labels[i])) for i in pick]


def _pick_one_test_image_per_class(meta: dict, seed: int) -> dict[str, str]:
    """One random absolute path per class label from the stratified test split."""
    paths = np.asarray(meta["test_paths"])
    labels = np.asarray(meta["test_labels"], dtype=np.int32)
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    for ci, cname in enumerate(config.CLASS_NAMES):
        idxs = np.where(labels == ci)[0]
        if len(idxs) == 0:
            raise ValueError(f"No test samples for class {cname}.")
        pick = int(rng.choice(idxs))
        out[cname] = str(paths[pick])
    return out


def _pick_gradcam_weights_slug(accs: dict[str, float]) -> str:
    """
    Prefer the Hybrid model when it matches the best test accuracy (ties included);
    otherwise use the strictly best-performing checkpoint for XAI.
    """
    best = max(accs.values())
    if accs.get("hybrid", -1.0) >= best - 1e-12:
        return "hybrid"
    return max(accs.items(), key=lambda kv: kv[1])[0]


def _load_model_for_gradcam(hp: dict) -> tf.keras.Model:
    model = model_builder.build_model(
        learning_rate=hp["learning_rate"],
        dropout_rate=hp["dropout"],
    )
    return model


def _run_gradcam_exports(
    xai_model: tf.keras.Model,
    meta_preview: dict,
    test_pick_seed: int | None = None,
) -> None:
    """Save one Grad-CAM figure per class using stratified test image paths."""
    seed = (
        test_pick_seed if test_pick_seed is not None else config.RANDOM_SEED + 777
    )
    per_class_paths = _pick_one_test_image_per_class(meta_preview, seed)
    for cname in config.CLASS_NAMES:
        out_png = config.OUTPUT_DIR / f"gradcam_{cname}.png"
        gradcam_mod.save_gradcam_for_path(
            per_class_paths[cname],
            xai_model,
            out_png,
            true_class_name=cname,
        )
        print(f"Saved Grad-CAM visualization to {out_png}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPN CNN + meta-heuristics + Grad-CAM (use --gradcam-only to skip training)."
    )
    parser.add_argument(
        "--gradcam-only",
        action="store_true",
        help="Load saved .weights.h5 and regenerate Grad-CAM PNGs only (no search or training).",
    )
    parser.add_argument(
        "--weights-slug",
        type=str,
        default="baseline",
        choices=("baseline", "pso", "gwo", "hybrid"),
        help="Stem for weights_<slug>.weights.h5 under the project root (ignored if --weights-path is set).",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Explicit path to a .weights.h5 file (overrides --weights-slug).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Adam LR used when the checkpoint was trained (must match for load_weights). "
        "Defaults to config baseline when --weights-slug baseline.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout on the classification head (must match training). "
        "Defaults to config baseline when --weights-slug baseline.",
    )
    parser.add_argument(
        "--gradcam-grid",
        action="store_true",
        help="Load weights and save one PNG: N random test rows × (input + Grad-CAM per class).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=5,
        help="Number of random test images for --gradcam-grid (default: 5).",
    )
    parser.add_argument(
        "--grid-seed",
        type=int,
        default=None,
        help="RNG seed for picking test images in --gradcam-grid (default: RANDOM_SEED+888).",
    )
    return parser.parse_args()


def _gradcam_cli_load_model_and_meta(
    args: argparse.Namespace,
) -> tuple[tf.keras.Model, dict, Path]:
    """Shared loader for ``--gradcam-only`` and ``--gradcam-grid``."""
    slug = args.weights_slug
    if slug != "baseline" and (args.learning_rate is None or args.dropout is None):
        raise SystemExit(
            "For --weights-slug pso|gwo|hybrid, you must pass --learning-rate and --dropout "
            "matching that training run so the model graph matches the saved weights."
        )
    lr = (
        args.learning_rate
        if args.learning_rate is not None
        else config.BASELINE_LR
    )
    dr = args.dropout if args.dropout is not None else config.BASELINE_DROPOUT
    wpath = args.weights_path if args.weights_path is not None else _weights_path(slug)
    if not wpath.is_file():
        raise SystemExit(f"Weights file not found: {wpath}")

    print(f"Backbone: {config.BACKBONE}")
    print(f"Dataset root: {config.DATASET_DIR}")
    print(f"[Grad-CAM] Loading weights from {wpath}")
    _, _, _, meta_preview = data_mod.prepare_datasets(
        batch_size=config.BASELINE_BATCH, seed=config.RANDOM_SEED
    )
    print(
        f"Splits — train: {meta_preview['n_train']}, "
        f"val: {meta_preview['n_val']}, test: {meta_preview['n_test']}"
    )

    hp = _hp_from_dict(lr, dr, config.BASELINE_BATCH)
    xai_model = _load_model_for_gradcam(hp)
    xai_model.load_weights(str(wpath))
    return xai_model, meta_preview, wpath


def _run_gradcam_only_cli(args: argparse.Namespace) -> None:
    set_global_seeds(config.RANDOM_SEED)
    xai_model, meta_preview, _wpath = _gradcam_cli_load_model_and_meta(args)
    _run_gradcam_exports(xai_model, meta_preview)
    del xai_model
    tf.keras.backend.clear_session()
    print("\nDone (Grad-CAM only).")


def _run_gradcam_grid_cli(args: argparse.Namespace) -> None:
    set_global_seeds(config.RANDOM_SEED)
    xai_model, meta_preview, _wpath = _gradcam_cli_load_model_and_meta(args)
    grid_seed = (
        args.grid_seed if args.grid_seed is not None else config.RANDOM_SEED + 888
    )
    rows = _pick_n_random_test_samples(meta_preview, args.grid_rows, grid_seed)
    out_png = (
        config.OUTPUT_DIR
        / f"gradcam_grid_{args.grid_rows}x{len(config.CLASS_NAMES)}classes_jet.png"
    )
    gradcam_mod.save_gradcam_per_class_grid(xai_model, rows, out_png)
    print(f"Saved multi-input / per-class Grad-CAM grid to {out_png}")
    del xai_model
    tf.keras.backend.clear_session()
    print("\nDone (Grad-CAM grid).")


def main() -> None:
    args = _parse_args()
    if args.gradcam_grid:
        _run_gradcam_grid_cli(args)
        return
    if args.gradcam_only:
        _run_gradcam_only_cli(args)
        return

    set_global_seeds(config.RANDOM_SEED)

    print(f"Backbone: {config.BACKBONE}")
    print(f"Dataset root: {config.DATASET_DIR}")
    _, _, _, meta_preview = data_mod.prepare_datasets(
        batch_size=config.BASELINE_BATCH, seed=config.RANDOM_SEED
    )
    print(
        f"Splits — train: {meta_preview['n_train']}, "
        f"val: {meta_preview['n_val']}, test: {meta_preview['n_test']}"
    )

    # --- Meta-heuristic search (validation loss fitness) ---
    print("\n[PSO] Running hyperparameter search...")
    pso_best, pso_trace = meta_heuristics.optimize_pso(verbose=0)
    print(f"[PSO] Best hyperparameters: {pso_best}")
    print(f"[PSO] Global-best val loss trace: {pso_trace}")

    print("\n[GWO] Running hyperparameter search...")
    gwo_best, gwo_trace = meta_heuristics.optimize_gwo(verbose=0)
    print(f"[GWO] Best hyperparameters: {gwo_best}")
    print(f"[GWO] Global-best val loss trace: {gwo_trace}")

    print("\n[Hybrid PSO-GWO] Running hyperparameter search...")
    hybrid_best, hybrid_trace = meta_heuristics.optimize_hybrid_pso_gwo(verbose=0)
    print(f"[Hybrid] Best hyperparameters: {hybrid_best}")
    print(f"[Hybrid] Global-best val loss trace: {hybrid_trace}")

    tf.keras.backend.clear_session()

    # --- Final full training (four separate models) ---
    baseline_hp = _hp_from_dict(
        config.BASELINE_LR, config.BASELINE_DROPOUT, config.BASELINE_BATCH
    )
    pso_hp = _hp_from_dict(
        pso_best["learning_rate"], pso_best["dropout"], pso_best["batch_size"]
    )
    gwo_hp = _hp_from_dict(
        gwo_best["learning_rate"], gwo_best["dropout"], gwo_best["batch_size"]
    )
    hybrid_hp = _hp_from_dict(
        hybrid_best["learning_rate"],
        hybrid_best["dropout"],
        hybrid_best["batch_size"],
    )

    print("\n=== Final training: Baseline ===")
    h_base, y_true_b, proba_base, _ = train_evaluate_full(
        "Baseline",
        **baseline_hp,
        epochs=config.FINAL_EPOCHS,
        weights_slug="baseline",
        verbose_fit=1,
    )

    print("\n=== Final training: PSO-optimized ===")
    h_pso, y_true_p, proba_pso, _ = train_evaluate_full(
        "PSO",
        **pso_hp,
        epochs=config.FINAL_EPOCHS,
        weights_slug="pso",
        verbose_fit=1,
    )

    print("\n=== Final training: GWO-optimized ===")
    h_gwo, y_true_g, proba_gwo, _ = train_evaluate_full(
        "GWO",
        **gwo_hp,
        epochs=config.FINAL_EPOCHS,
        weights_slug="gwo",
        verbose_fit=1,
    )

    print("\n=== Final training: Hybrid PSO-GWO ===")
    h_hyb, y_true_h, proba_hyb, _ = train_evaluate_full(
        "Hybrid",
        **hybrid_hp,
        epochs=config.FINAL_EPOCHS,
        weights_slug="hybrid",
        verbose_fit=1,
    )

    assert (
        np.array_equal(y_true_b, y_true_p)
        and np.array_equal(y_true_b, y_true_g)
        and np.array_equal(y_true_b, y_true_h)
    )
    y_true = y_true_b

    pred_base = np.argmax(proba_base, axis=1)
    pred_pso = np.argmax(proba_pso, axis=1)
    pred_gwo = np.argmax(proba_gwo, axis=1)
    pred_hyb = np.argmax(proba_hyb, axis=1)

    rows = [
        paper_outputs.classification_report_row("Baseline CNN", y_true, pred_base),
        paper_outputs.classification_report_row("CNN + PSO", y_true, pred_pso),
        paper_outputs.classification_report_row("CNN + GWO", y_true, pred_gwo),
        paper_outputs.classification_report_row(
            "CNN + Hybrid PSO-GWO", y_true, pred_hyb
        ),
    ]
    df = paper_outputs.build_comparison_dataframe(rows)
    paper_outputs.print_comparison_table(df)
    csv_path = paper_outputs.save_comparison_csv(df)
    print(f"Saved metrics table to {csv_path}")

    roc_path = paper_outputs.plot_combined_roc(
        y_true,
        {
            "Baseline CNN": proba_base,
            "CNN + PSO": proba_pso,
            "CNN + GWO": proba_gwo,
            "CNN + Hybrid PSO-GWO": proba_hyb,
        },
    )
    print(f"Saved ROC overlay to {roc_path}")

    curves_path = paper_outputs.plot_combined_training_histories(
        {
            "Baseline": h_base,
            "PSO": h_pso,
            "GWO": h_gwo,
            "Hybrid": h_hyb,
        }
    )
    print(f"Saved training curves to {curves_path}")

    for title, preds, slug in [
        ("Baseline CNN", pred_base, "baseline"),
        ("CNN + PSO", pred_pso, "pso"),
        ("CNN + GWO", pred_gwo, "gwo"),
        ("CNN + Hybrid PSO-GWO", pred_hyb, "hybrid"),
    ]:
        p = paper_outputs.save_confusion_matrix_figure(
            y_true, preds, title, filename_slug=slug
        )
        print(f"Saved confusion matrix to {p}")

    # --- Grad-CAM (XAI): one random test image per class ---
    accs = {
        "baseline": float(accuracy_score(y_true, pred_base)),
        "pso": float(accuracy_score(y_true, pred_pso)),
        "gwo": float(accuracy_score(y_true, pred_gwo)),
        "hybrid": float(accuracy_score(y_true, pred_hyb)),
    }
    gradcam_slug = _pick_gradcam_weights_slug(accs)
    hp_by_slug = {
        "baseline": baseline_hp,
        "pso": pso_hp,
        "gwo": gwo_hp,
        "hybrid": hybrid_hp,
    }
    print(
        f"\n[Grad-CAM] Using '{gradcam_slug}' weights "
        f"(test accuracies: {accs}); colormap='jet'."
    )

    xai_model = _load_model_for_gradcam(hp_by_slug[gradcam_slug])
    xai_model.load_weights(str(_weights_path(gradcam_slug)))

    _run_gradcam_exports(xai_model, meta_preview)

    del xai_model
    tf.keras.backend.clear_session()

    print("\nDone.")


if __name__ == "__main__":
    main()
