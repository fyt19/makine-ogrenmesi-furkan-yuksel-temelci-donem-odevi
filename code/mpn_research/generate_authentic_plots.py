#!/usr/bin/env python3
"""
Evaluate saved checkpoints on the real test split and export separate 300-DPI figures.

- Loads ``test_ds`` via ``mpn_research.data.prepare_datasets`` (class order: config.CLASS_NAMES = PV, ET, MF).
- Discovers only existing ``weights_<slug>.weights.h5`` files under the project root.
- Optional ``authentic_hparams.json`` in the project root supplies per-slug ``learning_rate`` / ``dropout``
  so PSO/GWO/Hybrid heads match saved weights (baseline defaults come from ``config``).

Does **not** generate Grad-CAM (assumed already on disk).

Run from ``code/`` directory::

    cd code
    python3 mpn_research/generate_authentic_plots.py

or::

    python3 -m mpn_research.generate_authentic_plots

Optional hyperparameters file (example)::

    {
      "pso": {"learning_rate": 0.0038, "dropout": 0.44},
      "gwo": {"learning_rate": 0.00136, "dropout": 0.38},
      "hybrid": {"learning_rate": 0.00355, "dropout": 0.57}
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Standalone: allow `python3 mpn_research/generate_authentic_plots.py` from project root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from mpn_research import config
from mpn_research import data as data_mod
from mpn_research import model_builder


WEIGHT_SLUGS = ("baseline", "pso", "gwo", "hybrid")

DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Baseline CNN",
    "pso": "CNN + PSO",
    "gwo": "CNN + GWO",
    "hybrid": "CNN + Hybrid PSO-GWO",
}

HP_JSON_NAME = "authentic_hparams.json"
OUT_METRICS_BAR = "authentic_metrics_bar.png"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _weights_path(slug: str) -> Path:
    return _project_root() / "çıktılar" / f"weights_{slug}.weights.h5"


def _load_hp_overrides() -> dict[str, dict[str, float]]:
    path = _project_root() / HP_JSON_NAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] Could not read {path}: {e}", file=sys.stderr)
        return {}
    out: dict[str, dict[str, float]] = {}
    if not isinstance(raw, dict):
        return {}
    for slug, body in raw.items():
        if not isinstance(body, dict):
            continue
        try:
            lr = float(body["learning_rate"]) if "learning_rate" in body else None
            dr = float(body["dropout"]) if "dropout" in body else None
        except (TypeError, ValueError):
            continue
        d: dict[str, float] = {}
        if lr is not None:
            d["learning_rate"] = lr
        if dr is not None:
            d["dropout"] = dr
        if d:
            out[str(slug)] = d
    return out


def _hp_for_slug(slug: str, overrides: dict[str, dict[str, float]]) -> dict[str, float]:
    lr = float(config.BASELINE_LR)
    dr = float(config.BASELINE_DROPOUT)
    if slug in overrides:
        lr = float(overrides[slug].get("learning_rate", lr))
        dr = float(overrides[slug].get("dropout", dr))
    return {"learning_rate": lr, "dropout": dr, "batch_size": int(config.BASELINE_BATCH)}


def _macro_specificity_from_cm(cm: np.ndarray) -> float:
    """Macro-averaged one-vs-rest specificity from a multi-class confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    n = cm.shape[0]
    specs: list[float] = []
    for c in range(n):
        tp = cm[c, c]
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        den = tn + fp
        specs.append((tn / den) if den > 0 else 0.0)
    return float(np.mean(specs))


def _collect_predictions(
    model: tf.keras.Model, test_ds: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    p_parts: list[np.ndarray] = []
    try:
        for images, labels in test_ds:
            y_parts.append(np.asarray(labels.numpy(), dtype=np.int32))
            p_parts.append(np.asarray(model.predict(images, verbose=0), dtype=np.float64))
    except Exception as e:
        raise RuntimeError(f"Prediction loop failed: {e}") from e
    y_true = np.concatenate(y_parts, axis=0)
    proba = np.concatenate(p_parts, axis=0)
    return y_true, proba


def _metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "specificity": _macro_specificity_from_cm(cm),
        "cm": cm,
    }


def _setup_plot_style() -> None:
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.95)
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def _save_metrics_bar(
    slugs: list[str],
    metrics_by_slug: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    _setup_plot_style()
    display = [DISPLAY_NAMES.get(s, s) for s in slugs]
    acc = np.array([metrics_by_slug[s]["accuracy"] for s in slugs], dtype=np.float64)
    f1v = np.array([metrics_by_slug[s]["f1"] for s in slugs], dtype=np.float64)
    rec = np.array([metrics_by_slug[s]["recall"] for s in slugs], dtype=np.float64)
    spe = np.array([metrics_by_slug[s]["specificity"] for s in slugs], dtype=np.float64)

    mat = np.vstack([acc, f1v, rec, spe]).T
    names = ["Doğruluk", "F1", "Duyarlılık", "Özgüllük"]
    x = np.arange(len(slugs), dtype=np.float64)
    width = min(0.22, 0.8 / (len(names) + 1))
    offsets = np.linspace(-(len(names) - 1) / 2 * width, (len(names) - 1) / 2 * width, len(names))
    colors = sns.color_palette("muted", n_colors=len(names))

    fig, ax = plt.subplots(figsize=(max(9.0, 2.2 * len(slugs)), 6.0), constrained_layout=True)
    for j, (nm, off, c) in enumerate(zip(names, offsets, colors)):
        ax.bar(x + off, mat[:, j], width, label=nm, color=c, edgecolor="0.25", linewidth=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=12, ha="right")
    ax.set_ylabel("Skor (makro)", fontsize=13)
    ax.set_title(
        f"Gerçek test metrikleri — {config.BACKBONE} (dpi=300)",
        fontsize=15,
        pad=12,
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    try:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def _save_confusion_png(cm: np.ndarray, title: str, out_path: Path) -> None:
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    try:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            vmin=0,
            vmax=max(int(cm.max()), 1),
            square=True,
            linewidths=0.75,
            linecolor="white",
            cbar_kws={"shrink": 0.82, "label": "Örnek sayısı"},
            xticklabels=list(config.CLASS_NAMES),
            yticklabels=list(config.CLASS_NAMES),
            ax=ax,
        )
        ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=13)
        ax.set_ylabel("Gerçek Sınıf", fontsize=13)
        ax.set_title(title, fontsize=15, pad=12)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def main() -> int:
    root = config.OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {root}")
    print(f"Backbone (config): {config.BACKBONE}")

    overrides = _load_hp_overrides()
    if overrides:
        print(f"Loaded hyperparameter overrides from {HP_JSON_NAME}: {sorted(overrides.keys())}")
    else:
        print(
            f"[info] No {HP_JSON_NAME} found; using config baseline LR/dropout for all slugs "
            "(PSO/GWO/Hybrid loads may fail unless dropout matches training)."
        )

    try:
        _, _, test_ds, meta = data_mod.prepare_datasets(
            batch_size=config.BASELINE_BATCH, seed=config.RANDOM_SEED
        )
    except Exception as e:
        print(f"[error] Could not build datasets: {e}", file=sys.stderr)
        return 1

    print(
        f"Test split size (meta): n_test={meta.get('n_test', '?')} "
        f"(classes {list(config.CLASS_NAMES)})"
    )

    existing: list[str] = []
    for slug in WEIGHT_SLUGS:
        wp = _weights_path(slug)
        if wp.is_file():
            existing.append(slug)
        else:
            print(f"[skip] Missing weights: {wp.name}")

    if not existing:
        print("[error] No weight files found; nothing to evaluate.", file=sys.stderr)
        return 2

    metrics_by_slug: dict[str, dict[str, float]] = {}

    for slug in existing:
        wpath = _weights_path(slug)
        hp = _hp_for_slug(slug, overrides)
        print(f"\n== {slug} ==\n  weights: {wpath.name}\n  hp: lr={hp['learning_rate']}, dropout={hp['dropout']}")
        try:
            tf.keras.backend.clear_session()
            model = model_builder.build_model(
                learning_rate=hp["learning_rate"],
                dropout_rate=hp["dropout"],
            )
            model.load_weights(str(wpath))
        except Exception as e:
            print(f"[warn] Could not build/load model for {slug}: {e}", file=sys.stderr)
            continue

        try:
            y_true, proba = _collect_predictions(model, test_ds)
        except Exception as e:
            print(f"[warn] Prediction failed for {slug}: {e}", file=sys.stderr)
            del model
            tf.keras.backend.clear_session()
            continue

        y_pred = np.argmax(proba, axis=1)
        try:
            bundle = _metrics_bundle(y_true, y_pred)
        except Exception as e:
            print(f"[warn] Metric computation failed for {slug}: {e}", file=sys.stderr)
            del model
            tf.keras.backend.clear_session()
            continue

        metrics_by_slug[slug] = {
            "accuracy": bundle["accuracy"],
            "recall": bundle["recall"],
            "f1": bundle["f1"],
            "specificity": bundle["specificity"],
        }
        print(
            f"  accuracy={bundle['accuracy']:.4f}  recall={bundle['recall']:.4f}  "
            f"specificity={bundle['specificity']:.4f}  f1={bundle['f1']:.4f}"
        )

        cm_path = root / f"authentic_cm_{slug}.png"
        try:
            _save_confusion_png(
                bundle["cm"],
                f"Karmaşıklık matrisi — {DISPLAY_NAMES.get(slug, slug)}",
                cm_path,
            )
            print(f"  saved: {cm_path.name}")
        except Exception as e:
            print(f"[warn] Could not save CM figure for {slug}: {e}", file=sys.stderr)

        del model
        tf.keras.backend.clear_session()

    evaluated = [s for s in existing if s in metrics_by_slug]
    if not evaluated:
        print("[error] No models were successfully evaluated.", file=sys.stderr)
        return 3

    bar_path = root / OUT_METRICS_BAR
    try:
        _save_metrics_bar(evaluated, metrics_by_slug, bar_path)
        print(f"\nSaved metrics bar chart: {bar_path}")
    except Exception as e:
        print(f"[warn] Could not save metrics bar chart: {e}", file=sys.stderr)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
