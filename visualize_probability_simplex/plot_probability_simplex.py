"""Visualize definition probabilities against PCA projections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.collections import PolyCollection
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

DEFAULT_SCATTER_NAME = "{}_pca_scatter.png"
DEFAULT_KDE_NAME = "{}_pca_kde.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA of feature activations colored by definition probs")
    parser.add_argument("scored_path", type=Path, help="Parquet produced by score_definitions.py")
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=None,
        help="PNG output path for the colored scatter (default: <input>_pca_scatter.png)",
    )
    parser.add_argument(
        "--kde-output",
        type=Path,
        default=None,
        help="PNG output path for the density heatmap (default: <input>_pca_kde.png)",
    )
    parser.add_argument("--point-size", type=float, default=30.0, help="Marker size for scatter plot")
    parser.add_argument("--alpha", type=float, default=0.55, help="Marker opacity")
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=1.6,
        help="Bandwidth for the classwise KDE heatmaps",
    )
    parser.add_argument(
        "--kde-grid",
        type=int,
        default=90,
        help="Grid resolution for KDE heatmaps",
    )
    parser.add_argument(
        "--skip-kde",
        action="store_true",
        help="Skip generating the KDE heatmap figure",
    )
    return parser.parse_args()


def default_paths(base: Path) -> Tuple[Path, Path]:
    stem = base.stem
    parent = base.parent
    return (
        parent / DEFAULT_SCATTER_NAME.format(stem),
        parent / DEFAULT_KDE_NAME.format(stem),
    )


def load_metadata(table) -> Dict[str, str]:
    metadata = table.schema.metadata or {}
    return {key.decode(): value.decode() for key, value in metadata.items()}


def compute_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.shape[0] < 2:
        raise ValueError("At least two examples are required for PCA")
    components = min(matrix.shape[1], matrix.shape[0], 3)
    pca = PCA(n_components=components)
    pcs = pca.fit_transform(matrix)
    return pcs, pca.explained_variance_ratio_


def regression_r2(pcs: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if pcs.shape[1] < 2:
        raise ValueError("Need at least two principal components for regression analysis")
    X = pcs[:, :2]
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    design = np.concatenate([X, ones], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, probs, rcond=None)
    predictions = design @ coeffs
    residual = np.sum((probs - predictions) ** 2, axis=0)
    total = np.sum((probs - probs.mean(axis=0, keepdims=True)) ** 2, axis=0)
    overall_total = total.sum()
    overall_residual = residual.sum()

    def safe_ratio(num: float, denom: float) -> float:
        return float("nan") if denom <= 0 else 1.0 - num / denom

    r2_per = [safe_ratio(residual[i], total[i]) for i in range(probs.shape[1])]
    overall = safe_ratio(overall_residual, overall_total)
    return {
        "overall": overall,
        "per_label": r2_per,
    }


def _base_palette() -> np.ndarray:
    return np.array(
        [
            (1.0, 0.2, 0.2),
            (0.0, 0.6, 0.0),
            (0.2, 0.2, 1.0),
        ],
        dtype=np.float32,
    )


def _compute_bounds(pcs: np.ndarray, margin_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    x = pcs[:, 0]
    y = pcs[:, 1]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin_x = margin_ratio * (x_range if x_range > 0 else 1.0)
    margin_y = margin_ratio * (y_range if y_range > 0 else 1.0)
    return x.min() - margin_x, x.max() + margin_x, y.min() - margin_y, y.max() + margin_y


def _generate_quads(grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    x0 = grid_x[:-1, :-1]
    x1 = grid_x[1:, :-1]
    x2 = grid_x[1:, 1:]
    x3 = grid_x[:-1, 1:]

    y0 = grid_y[:-1, :-1]
    y1 = grid_y[1:, :-1]
    y2 = grid_y[1:, 1:]
    y3 = grid_y[:-1, 1:]

    quads = np.stack(
        [
            np.stack([x0, y0], axis=-1),
            np.stack([x1, y1], axis=-1),
            np.stack([x2, y2], axis=-1),
            np.stack([x3, y3], axis=-1),
        ],
        axis=-2,
    )
    return quads.reshape(-1, 4, 2)


def plot_scatter(
    pcs: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    output_path: Path,
    point_size: float,
    alpha: float,
    variance_ratio: np.ndarray,
) -> None:
    base_colors = _base_palette()
    colors = np.clip(probs @ base_colors, 0.0, 1.0)
    x_min, x_max, y_min, y_max = _compute_bounds(pcs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=point_size, alpha=alpha, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.2%} var)" if variance_ratio.size > 1 else "PC2")
    ax.set_title("Definition probability simplex – scatter")
    ax.set_aspect('auto')
    for label, color in zip(labels[:3], base_colors):
        ax.scatter([], [], c=[color], label=label)
    ax.legend(title="Definition", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_kde(
    pcs: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    output_path: Path,
    variance_ratio: np.ndarray,
    bandwidth: float,
    grid_size: int,
) -> None:
    base_colors = _base_palette()
    class_index = probs.argmax(axis=1)

    x_min, x_max, y_min, y_max = _compute_bounds(pcs)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, color in enumerate(base_colors):
        points = pcs[class_index == idx]
        if points.shape[0] < 5:
            continue
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(points[:, :2])
        density = np.exp(kde.score_samples(grid_points)).reshape(grid_x.shape)
        max_density = density.max()
        if max_density <= 0:
            continue
        density /= max_density
        alpha_map = np.clip(density, 0.0, 1.0) * 0.6

        quads = _generate_quads(grid_x, grid_y)
        colors_rgba = np.zeros((quads.shape[0], 4), dtype=np.float32)
        colors_rgba[:, :3] = color
        colors_rgba[:, 3] = alpha_map[:-1, :-1].ravel()

        collection = PolyCollection(quads, facecolors=colors_rgba, edgecolors='none')
        ax.add_collection(collection)
        ax.scatter([], [], c=[color], label=labels[idx])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.2%} var)" if variance_ratio.size > 1 else "PC2")
    ax.set_title("Definition probability simplex – KDE heatmap")
    ax.legend(title="Definition", loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.scored_path)
    metadata = load_metadata(table)

    activation_vectors = table.column("selected_activation_vector").to_pylist()
    matrix = np.asarray(activation_vectors, dtype=np.float32)

    probabilities = table.column("definition_probabilities").to_pylist()
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape[1] != 3:
        raise ValueError("Plotting currently expects exactly three definition choices")

    pcs, variance_ratio = compute_pca(matrix)
    stats = regression_r2(pcs, probs)

    labels = json.loads(metadata.get("choice_labels", "[]"))
    if len(labels) != probs.shape[1]:
        labels = [f"Choice {i}" for i in range(probs.shape[1])]

    scatter_path, kde_path_default = default_paths(args.scored_path)
    scatter_output = args.scatter_output or scatter_path
    kde_output = args.kde_output or kde_path_default

    plot_scatter(pcs, probs, labels, scatter_output, args.point_size, args.alpha, variance_ratio)
    print(f"Saved PCA scatter to {scatter_output}")

    if not args.skip_kde:
        plot_kde(
            pcs[:, :2],
            probs,
            labels,
            kde_output,
            variance_ratio,
            args.kde_bandwidth,
            args.kde_grid,
        )
        print(f"Saved PCA KDE to {kde_output}")

    per_label_r2 = {label: value for label, value in zip(labels, stats["per_label"]) }
    print(f"Overall R^2 (PC1/PC2 -> probabilities): {stats['overall']:.4f}")
    for label in labels:
        value = per_label_r2.get(label, float("nan"))
        print(f"  {label}: {value:.4f}")


if __name__ == "__main__":
    main()
