from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configuration
CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
PROBE_PATH = Path("models/linear_cone_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCA_DIM = 1024  # per-layer components in cache

TOP_N_PAIRS = 8
MIN_TOTAL_MASS = 0.1  # minimum combined probability for both classes
PRESENCE_THRESH = 0.0  # for Jaccard: label present if prob > thresh
GRID_COLS = 4  # 4 columns x 2 rows for 8 pairs

# Output files
SIMPLEX_FIG = Path("models/top_1_simplexes_simplex.png")
RESIDUAL_FIG = Path("models/top_1_simplexes_residual.png")
HEATMAP_FIG = Path("models/top_1_simplexes_heatmap.png")
CONNECTED_FIG = Path("models/top_1_simplexes_connected.png")


class LinearConeProbe(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        cone_coords = torch.relu(self.linear(x))
        return cone_coords / (cone_coords.sum(dim=-1, keepdim=True) + 1e-8)


def load_cache(cache_base: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load cached activations and labels."""
    x_path = cache_base.with_name(cache_base.name + "_X.npy")
    y_path = cache_base.with_name(cache_base.name + "_Y.npy")
    meta_path = cache_base.with_name(cache_base.name + "_meta.npz")
    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Cache files missing for base {cache_base}")
    meta = np.load(meta_path, allow_pickle=True)
    tags = meta.get("tag_names")
    layers = meta.get("layers")
    if tags is None or layers is None:
        raise ValueError(f"Cache {cache_base} missing tag_names or layers.")
    if list(layers) != LAYERS:
        raise ValueError(f"Layer mismatch in cache {cache_base}; expected {LAYERS}, found {list(layers)}")
    X = np.load(x_path).astype(np.float32)
    Y = np.load(y_path).astype(np.float32)
    return X, Y, list(tags)


def load_probe(path: Path, hidden_dim: int, num_classes: int) -> LinearConeProbe:
    """Load trained LinearConeProbe."""
    state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = LinearConeProbe(hidden_dim, num_classes)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    return model.eval().cuda() if torch.cuda.is_available() else model.eval()


def align_distributions(Y: np.ndarray, source_tags: List[str], target_tags: List[str]) -> np.ndarray:
    """Align label distributions to target tag order."""
    idx = {t: i for i, t in enumerate(source_tags)}
    cols = [idx[t] for t in target_tags if t in idx]
    aligned = Y[:, cols]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned / row_sums


def compute_jaccard(Y: np.ndarray) -> np.ndarray:
    """Compute Jaccard similarity matrix for tag co-occurrence."""
    B = (Y > PRESENCE_THRESH).astype(np.float32)
    counts = B.sum(axis=0)
    intersect = B.T @ B
    union = counts[:, None] + counts[None, :] - intersect
    union = np.where(union == 0, 1.0, union)
    return intersect / union


def get_top_pairs(jacc: np.ndarray, tag_names: List[str], n: int) -> List[Tuple[float, str, str, int, int]]:
    """Return top N pairs by Jaccard, excluding self-pairs."""
    pairs = []
    num_tags = len(tag_names)
    for i in range(num_tags):
        for j in range(i + 1, num_tags):
            pairs.append((jacc[i, j], tag_names[i], tag_names[j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:n]


def compute_1simplex_positions(
    gt: np.ndarray, pred: np.ndarray, idx_a: int, idx_b: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized 1-simplex positions for a pair of classes.
    Returns (gt_positions, pred_positions) filtered to samples with sufficient mass.
    """
    mass_a_gt = gt[:, idx_a]
    mass_b_gt = gt[:, idx_b]
    total_gt = mass_a_gt + mass_b_gt

    mass_a_pred = pred[:, idx_a]
    mass_b_pred = pred[:, idx_b]
    total_pred = mass_a_pred + mass_b_pred

    # Filter: require minimum mass in both gt and pred
    mask = (total_gt > MIN_TOTAL_MASS) & (total_pred > MIN_TOTAL_MASS)

    gt_pos = mass_a_gt[mask] / total_gt[mask]
    pred_pos = mass_a_pred[mask] / total_pred[mask]

    return gt_pos, pred_pos


def plot_simplex_grid(pairs_data: List[dict], output_path: Path):
    """Create grid of simple 1-simplex line plots: points on a line, colored by GT."""
    n_pairs = len(pairs_data)
    n_cols = GRID_COLS
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    for idx, data in enumerate(pairs_data):
        ax = axes[idx]
        gt_pos = data["gt_pos"]
        pred_pos = data["pred_pos"]

        # Two rows: GT on bottom (y=0), Pred on top (y=1)
        # Add small jitter to y for visibility
        rng = np.random.default_rng(42)
        jitter_gt = rng.uniform(-0.08, 0.08, len(gt_pos))
        jitter_pred = rng.uniform(-0.08, 0.08, len(pred_pos))

        # Plot GT points (bottom row)
        ax.scatter(
            gt_pos, jitter_gt,
            c=gt_pos, cmap="gist_rainbow", s=4, alpha=1.0, vmin=0, vmax=1,
            edgecolors="none"
        )
        # Plot Pred points (top row)
        ax.scatter(
            pred_pos, 1 + jitter_pred,
            c=gt_pos, cmap="gist_rainbow", s=4, alpha=1.0, vmin=0, vmax=1,
            edgecolors="none"
        )

        # Draw the 1-simplex lines
        ax.axhline(y=0, color="black", linestyle="-", linewidth=2)
        ax.axhline(y=1, color="black", linestyle="-", linewidth=2)

        # Endpoint labels
        ax.text(0, -0.25, data["tag_b"], fontsize=8, ha="center", va="top")
        ax.text(1, -0.25, data["tag_a"], fontsize=8, ha="center", va="top")
        ax.text(0, 1.25, data["tag_b"], fontsize=8, ha="center", va="bottom")
        ax.text(1, 1.25, data["tag_a"], fontsize=8, ha="center", va="bottom")

        # Row labels
        ax.text(-0.08, 0, "GT", fontsize=8, va="center", ha="right", fontweight="bold")
        ax.text(-0.08, 1, "Pred", fontsize=8, va="center", ha="right", fontweight="bold")

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(
            f"{data['tag_a']} vs {data['tag_b']} (J={data['jaccard']:.2f}, n={data['n_samples']})",
            fontsize=9
        )

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved simplex plots to {output_path}")


def plot_residual_grid(pairs_data: List[dict], output_path: Path):
    """Create grid of residual plots: x=gt_pos, y=(pred-gt), color=gt_pos."""
    n_pairs = len(pairs_data)
    n_cols = GRID_COLS
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    for idx, data in enumerate(pairs_data):
        ax = axes[idx]
        gt_pos = data["gt_pos"]
        pred_pos = data["pred_pos"]
        error = pred_pos - gt_pos

        scatter = ax.scatter(
            gt_pos, error,
            c=gt_pos, cmap="gist_rainbow", s=4, alpha=1.0, vmin=0, vmax=1,
            edgecolors="none"
        )
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"GT: {data['tag_a']} fraction", fontsize=8)
        ax.set_ylabel("Pred - GT", fontsize=8)

        mae = np.mean(np.abs(error))
        ax.set_title(
            f"{data['tag_a']} vs {data['tag_b']}\nJ={data['jaccard']:.2f}, n={data['n_samples']}, MAE={mae:.3f}",
            fontsize=9
        )

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual plots to {output_path}")


def plot_heatmap_grid(pairs_data: List[dict], output_path: Path):
    """Create grid of calibration heatmaps: 2D histogram of (gt_pos, pred_pos)."""
    n_pairs = len(pairs_data)
    n_cols = GRID_COLS
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    bins = np.linspace(0, 1, 21)

    for idx, data in enumerate(pairs_data):
        ax = axes[idx]
        gt_pos = data["gt_pos"]
        pred_pos = data["pred_pos"]

        h, xedges, yedges = np.histogram2d(gt_pos, pred_pos, bins=bins)
        h = np.log1p(h)  # log scale for visibility

        im = ax.imshow(
            h.T, origin="lower", extent=[0, 1, 0, 1],
            cmap="viridis", aspect="equal"
        )
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, alpha=0.8, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"GT: {data['tag_a']} fraction", fontsize=8)
        ax.set_ylabel(f"Pred: {data['tag_a']} fraction", fontsize=8)
        ax.set_title(
            f"{data['tag_a']} vs {data['tag_b']} (J={data['jaccard']:.2f})",
            fontsize=9
        )

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap plots to {output_path}")


def plot_connected_grid(pairs_data: List[dict], output_path: Path):
    """Create grid of connected dot plots: lines connecting gt to pred positions."""
    n_pairs = len(pairs_data)
    n_cols = GRID_COLS
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    max_points = 300  # subsample for clarity

    for idx, data in enumerate(pairs_data):
        ax = axes[idx]
        gt_pos = data["gt_pos"]
        pred_pos = data["pred_pos"]

        # Subsample if too many points
        if len(gt_pos) > max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(gt_pos), max_points, replace=False)
            gt_pos = gt_pos[indices]
            pred_pos = pred_pos[indices]

        # Draw connecting lines
        for g, p in zip(gt_pos, pred_pos):
            ax.plot([g, p], [0, 1], color="gray", alpha=0.15, linewidth=0.5)

        # Draw points on both tracks
        ax.scatter(gt_pos, np.zeros_like(gt_pos), c=gt_pos, cmap="gist_rainbow",
                   s=4, alpha=1.0, vmin=0, vmax=1, zorder=5, edgecolors="none")
        ax.scatter(pred_pos, np.ones_like(pred_pos), c=gt_pos, cmap="gist_rainbow",
                   s=4, alpha=1.0, vmin=0, vmax=1, zorder=5, edgecolors="none")

        # Track labels
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.axhline(y=1, color="black", linestyle="-", linewidth=1, alpha=0.5)
        ax.text(-0.08, 0, "GT", fontsize=8, va="center", ha="right")
        ax.text(-0.08, 1, "Pred", fontsize=8, va="center", ha="right")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel(f"{data['tag_a']} fraction", fontsize=8)
        ax.set_yticks([])
        ax.set_title(
            f"{data['tag_a']} vs {data['tag_b']} (J={data['jaccard']:.2f})",
            fontsize=9
        )

    # Hide unused axes
    for ax in axes[n_pairs:]:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved connected plots to {output_path}")


def main():
    # 1. Load cache and probe tag order
    print("Loading cache...")
    X, Y, cache_tags = load_cache(CACHE_BASE)

    if TAG_NAMES_PATH.exists():
        probe_tags = json.load(TAG_NAMES_PATH.open())
    else:
        raise ValueError(f"Probe tag list not found at {TAG_NAMES_PATH}")

    # Align cache labels to probe tag order
    missing = [t for t in probe_tags if t not in cache_tags]
    if missing:
        raise ValueError(f"Cache missing probe tags: {missing}")
    Y = align_distributions(Y, cache_tags, probe_tags)
    tag_names = probe_tags

    # 2. Load probe
    print("Loading probe...")
    probe = load_probe(PROBE_PATH, X.shape[1], len(tag_names))

    # 3. Run inference
    print("Running inference...")
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)

    preds = []
    device = next(probe.parameters()).device
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            pred = probe(batch_x)
            preds.append(pred.cpu().numpy())
    pred = np.concatenate(preds, axis=0)

    # 4. Compute Jaccard and get top pairs
    print("Computing Jaccard similarity...")
    jacc = compute_jaccard(Y)
    top_pairs = get_top_pairs(jacc, tag_names, TOP_N_PAIRS)

    print(f"\nTop {TOP_N_PAIRS} pairs by Jaccard:")
    for score, tag_a, tag_b, idx_a, idx_b in top_pairs:
        print(f"  {tag_a} vs {tag_b}: J={score:.3f}")

    # 5. Compute 1-simplex data for each pair
    pairs_data = []
    for score, tag_a, tag_b, idx_a, idx_b in top_pairs:
        gt_pos, pred_pos = compute_1simplex_positions(Y, pred, idx_a, idx_b)
        pairs_data.append({
            "tag_a": tag_a,
            "tag_b": tag_b,
            "jaccard": score,
            "gt_pos": gt_pos,
            "pred_pos": pred_pos,
            "n_samples": len(gt_pos),
        })
        print(f"  {tag_a} vs {tag_b}: {len(gt_pos)} samples after filtering")

    # 6. Generate all four visualization styles
    print("\nGenerating visualizations...")
    plot_simplex_grid(pairs_data, SIMPLEX_FIG)
    plot_residual_grid(pairs_data, RESIDUAL_FIG)
    plot_heatmap_grid(pairs_data, HEATMAP_FIG)
    plot_connected_grid(pairs_data, CONNECTED_FIG)

    print("\nDone!")


if __name__ == "__main__":
    main()
