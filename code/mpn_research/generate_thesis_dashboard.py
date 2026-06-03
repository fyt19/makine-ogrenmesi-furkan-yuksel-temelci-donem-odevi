#!/usr/bin/env python3
"""
Standalone thesis-quality figures (no model retraining, no CSV reads).

Writes **four separate** 300-DPI PNGs (one panel each) for Word / dissertation.

Run from ``code/`` directory:

    cd code
    python3 mpn_research/generate_thesis_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Hardcoded experiment results (thesis figures)
# ---------------------------------------------------------------------------
MODEL_LABELS = [
    "Baseline ResNet50",
    "CNN + PSO",
    "CNN + GWO",
    "Hybrid PSO-GWO",
]

ACCURACY = np.array([0.9833, 0.9667, 0.9500, 0.9500], dtype=np.float64)
F1_SCORE = np.array([0.9833, 0.9667, 0.9499, 0.9499], dtype=np.float64)
RECALL = np.array([0.9833, 0.9667, 0.9500, 0.9500], dtype=np.float64)
SPECIFICITY = np.array([0.9917, 0.9833, 0.9750, 0.9750], dtype=np.float64)

CLASS_NAMES = ["PV", "ET", "MF"]

CM_BASELINE = np.array([[20, 0, 0], [1, 19, 0], [0, 0, 20]], dtype=np.int32)
CM_HYBRID = np.array([[19, 1, 0], [2, 18, 0], [0, 0, 20]], dtype=np.int32)

PLACEHOLDER_TR = (
    "Lütfen bu alana Şekil 4.2'deki Grad-CAM Isı Haritalarını "
    "(Standart vs. Hibrit kıyası) manuel olarak ekleyiniz."
)

# Four standalone outputs (project root)
OUTPUT_FILES = (
    "mpn_thesis_fig_a_metrics_bar.png",
    "mpn_thesis_fig_b_cm_baseline.png",
    "mpn_thesis_fig_c_cm_hybrid.png",
    "mpn_thesis_fig_d_gradcam_placeholder.png",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir() -> Path:
    return _project_root() / "çıktılar"


def _setup_academic_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="muted")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def _plot_grouped_metrics(ax: plt.Axes) -> None:
    n_models = len(MODEL_LABELS)
    metrics = np.vstack([ACCURACY, F1_SCORE, RECALL, SPECIFICITY]).T
    metric_names = ["Doğruluk", "F1", "Duyarlılık", "Özgüllük"]
    x = np.arange(n_models, dtype=np.float64)
    width = 0.18
    offsets = np.linspace(-(1.5 * width), 1.5 * width, num=4)
    colors = sns.color_palette("muted", n_colors=4)

    for j, (mname, off, c) in enumerate(zip(metric_names, offsets, colors)):
        ax.bar(x + off, metrics[:, j], width, label=mname, color=c, edgecolor="0.25", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Baseline\nResNet50", "CNN +\nPSO", "CNN +\nGWO", "Hybrid\nPSO-GWO"],
        fontsize=10,
    )
    ax.set_ylabel("Skor (makro ortalama)", fontsize=13)
    ax.set_title(
        "Şekil (A) — Sınıflandırma metrikleri (model karşılaştırması)",
        fontsize=15,
        pad=14,
    )
    ax.set_ylim(0.90, 1.005)
    ax.legend(ncol=2, frameon=True, loc="lower right", fontsize=10)
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))


def _plot_confusion_heatmap(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        vmin=0,
        vmax=20,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.82, "label": "Örnek sayısı"},
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Tahmin edilen sınıf", fontsize=13)
    ax.set_ylabel("Gerçek sınıf", fontsize=13)
    ax.set_title(title, fontsize=15, pad=14)


def _plot_placeholder(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (0.05, 0.12),
            0.9,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#4c4c4c",
            facecolor="#fafafa",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.5,
        PLACEHOLDER_TR,
        ha="center",
        va="center",
        fontsize=13,
        wrap=True,
        linespacing=1.45,
        color="#222222",
        transform=ax.transAxes,
    )
    ax.set_title(
        "Şekil (D) — Kalitatif XAI alanı (Grad-CAM)",
        fontsize=15,
        pad=14,
    )


def generate_figure_files(out_dir: Path | None = None) -> list[Path]:
    """
    Save four 300-DPI PNGs under ``out_dir`` (default: project root).
    Returns paths in order A→D.
    """
    _setup_academic_style()
    root = out_dir or _output_dir()
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # (A) Grouped metrics
    fig_a, ax_a = plt.subplots(figsize=(11.0, 6.2), constrained_layout=True)
    _plot_grouped_metrics(ax_a)
    p_a = root / OUTPUT_FILES[0]
    fig_a.savefig(p_a, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)
    paths.append(p_a)

    # (B) Baseline CM
    fig_b, ax_b = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    _plot_confusion_heatmap(
        ax_b,
        CM_BASELINE,
        "Şekil (B) — Karmaşıklık matrisi: Baseline (ResNet50)",
    )
    p_b = root / OUTPUT_FILES[1]
    fig_b.savefig(p_b, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_b)
    paths.append(p_b)

    # (C) Hybrid CM
    fig_c, ax_c = plt.subplots(figsize=(6.8, 5.8), constrained_layout=True)
    _plot_confusion_heatmap(
        ax_c,
        CM_HYBRID,
        "Şekil (C) — Karmaşıklık matrisi: Hybrid PSO-GWO",
    )
    p_c = root / OUTPUT_FILES[2]
    fig_c.savefig(p_c, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_c)
    paths.append(p_c)

    # (D) Placeholder
    fig_d, ax_d = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    _plot_placeholder(ax_d)
    p_d = root / OUTPUT_FILES[3]
    fig_d.savefig(p_d, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_d)
    paths.append(p_d)

    return paths


def main() -> None:
    for p in generate_figure_files():
        print(f"Kaydedildi: {p}")


if __name__ == "__main__":
    main()
