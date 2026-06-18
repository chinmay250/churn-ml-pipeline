"""Model evaluation: scalar metrics + diagnostic plots.

Pure functions — no MLflow coupling here. ``train.py`` calls these and logs the
returned metrics/figure paths as MLflow metrics and artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Return the standard binary-classification metrics as a flat dict."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, title: str = ""
) -> Path:
    """Render and save a confusion-matrix PNG. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(title or "Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, out_path: str | Path, title: str = ""
) -> Path:
    """Render and save an ROC-curve PNG. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="#c44e52")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
