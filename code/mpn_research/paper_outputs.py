"""Test-set metrics, confusion matrices, and publication-style figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config


def macro_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None
) -> float:
    """One-vs-rest macro-averaged specificity from the confusion matrix."""
    labels = list(range(n_classes or config.NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specs: list[float] = []
    for c in labels:
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specs))


def classification_report_row(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )  # sensitivity / recall
    spec = macro_specificity(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Sensitivity (Recall)": sens,
        "Specificity": spec,
        "F1-Score": f1,
    }


def build_comparison_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric = [
        "Accuracy",
        "Sensitivity (Recall)",
        "Specificity",
        "F1-Score",
    ]
    df[numeric] = df[numeric].astype(float)
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty console table for thesis / log output."""
    disp = df.copy()
    for col in [
        "Accuracy",
        "Sensitivity (Recall)",
        "Specificity",
        "F1-Score",
    ]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Test-set comparison (macro-averaged where applicable) ===\n")
        print(disp.to_string(index=False))
        print()


def save_comparison_csv(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (config.OUTPUT_DIR / "comparison_metrics.csv")
    df.to_csv(path, index=False)
    return path


def plot_combined_roc(
    y_true: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    class_names: tuple[str, ...] | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Overlay micro-averaged multiclass ROC curves (one curve per model).
    """
    out_path = out_path or (config.OUTPUT_DIR / "combined_ROC.png")
    class_names = class_names or config.CLASS_NAMES
    classes = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(9, 7))
    for name, proba in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined micro-averaged ROC — Ph-negative MPN (3-class)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_combined_training_histories(
    histories: dict[str, dict],
    out_path: Path | None = None,
) -> Path:
    """Side-by-side loss and accuracy (train vs val) for all models."""
    out_path = out_path or (config.OUTPUT_DIR / "combined_training_curves.png")
    ncol = min(4, max(2, len(histories)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for name, h in histories.items():
        epochs = range(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["loss"], linestyle="--", label=f"{name} train")
        axes[0].plot(epochs, h["val_loss"], linestyle="-", label=f"{name} val")
        axes[1].plot(epochs, h["accuracy"], linestyle="--", label=f"{name} train")
        axes[1].plot(epochs, h["val_accuracy"], linestyle="-", label=f"{name} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Sparse categorical cross-entropy")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    h0, l0 = axes[0].get_legend_handles_labels()
    fig.legend(
        h0,
        l0,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fancybox=True,
        shadow=False,
        ncol=ncol,
        fontsize=8,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    filename_slug: str | None = None,
) -> Path:
    class_names = class_names or config.CLASS_NAMES
    out_dir = out_dir or config.OUTPUT_DIR
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"Confusion matrix — {model_name}")
    plt.tight_layout()
    safe = filename_slug or model_name.lower().replace("+", "plus").replace(" ", "_")
    path = out_dir / f"confusion_matrix_{safe}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return path
