"""Shared plotting helpers for probe evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def normalize_confusion_matrix(cm: np.ndarray, mode: str | None) -> np.ndarray:
    if mode is None:
        return cm
    cm = cm.astype(np.float64, copy=False)
    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    elif mode == "all":
        denom = np.full_like(cm, cm.sum())
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
    return normalized


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Iterable[str],
    path: Path,
    *,
    title: str,
    normalize: str | None = None,
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
) -> None:

    labels = list(labels)
    display_cm = normalize_confusion_matrix(cm, normalize)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display_cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True tag",
        xlabel="Predicted tag",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = display_cm.max() / 2 if display_cm.max() > 0 else 0.5
    for i in range(display_cm.shape[0]):
        for j in range(display_cm.shape[1]):
            value = display_cm[i, j]
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
