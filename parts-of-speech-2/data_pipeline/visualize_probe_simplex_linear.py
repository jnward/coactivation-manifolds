"""
Visualize simple linear probe predictions on a 2-simplex.

Unlike the cone probe visualization, this version does NOT normalize predictions
before barycentric projection. Predictions can land outside the triangle,
showing how far they deviate from valid probability space.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Configuration constants
CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
PROBE_PATH = Path("models/simple_linear_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
POS_CLASSES = ["ADV", "ADP", "SCONJ"]
SIMPLEX_FIG = Path("models/simplex_linear.png")
SCATTER_FIG = Path("models/pred_vs_gt_linear.png")
PRED_3D_HTML = Path("models/pred_linear_3d.html")
HEATMAP_FIG = Path("models/probe_direction_cosine_linear.png")
MIN_NONZERO_CLASSES = 2
MIN_TOTAL_MASS = 0.5
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCA_DIM = 1024


class SimpleLinearProbe(nn.Module):
    """Simple linear probe without ReLU or normalization."""
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # Direct linear output


def load_cache(cache_base: Path):
    """Load cached activations and labels."""
    x_path = cache_base.with_name(cache_base.name + "_X.npy")
    y_path = cache_base.with_name(cache_base.name + "_Y.npy")
    meta_path = cache_base.with_name(cache_base.name + "_meta.npz")

    X = np.load(x_path).astype(np.float32)
    Y = np.load(y_path).astype(np.float32)
    meta = np.load(meta_path, allow_pickle=True)
    tag_names = list(meta["tag_names"])

    return X, Y, tag_names


def load_tag_names(path: Path) -> List[str]:
    """Load tag names from JSON."""
    with open(path) as f:
        return json.load(f)


def load_probe(path: Path, hidden_dim: int, num_classes: int) -> SimpleLinearProbe:
    """Load simple linear probe."""
    state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLinearProbe(hidden_dim, num_classes)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    return model.eval().cuda()


def barycentric_to_xy(p1: float, p2: float, p3: float) -> Tuple[float, float]:
    """Convert barycentric coordinates to 2D xy coordinates on triangle."""
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2)
    x = p1 * A[0] + p2 * B[0] + p3 * C[0]
    y = p1 * A[1] + p2 * B[1] + p3 * C[1]
    return x, y


def align_distributions(Y: np.ndarray, source_tags: List[str], target_tags: List[str]) -> np.ndarray:
    """Select and renormalize columns to match target_tags ordering."""
    src_idx = {t: i for i, t in enumerate(source_tags)}
    cols = [src_idx[t] for t in target_tags if t in src_idx]
    aligned = Y[:, cols]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-8, None)
    return aligned / row_sums


def main():
    # Load tag names
    probe_tags = load_tag_names(TAG_NAMES_PATH)
    pos_indices = [probe_tags.index(pos) for pos in POS_CLASSES]
    print(f"Using POS classes: {POS_CLASSES} at indices {pos_indices}")

    # Load cached data
    print("Loading cached activations...")
    X, Y, cache_tags = load_cache(CACHE_BASE)
    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Cache tags: {cache_tags}")

    # Align if necessary
    if cache_tags != probe_tags:
        print("Aligning distributions to probe tag order...")
        Y = align_distributions(Y, cache_tags, probe_tags)

    # Create data loader
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # Load probe
    probe = load_probe(PROBE_PATH, X.shape[1], len(probe_tags))
    print(f"Loaded probe with weight shape: {probe.linear.weight.shape}")

    # Collect predictions
    pred_points = []  # Raw prediction barycentric points (may be outside triangle)
    gt_points = []    # Ground truth barycentric points (inside triangle)
    colors = []       # Colors based on GT
    raw_vectors = []
    activations = []
    per_class_gt = {tag: [] for tag in probe_tags}
    per_class_pred = {tag: [] for tag in probe_tags}

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            outputs = probe(batch_x)  # Raw linear output (no ReLU, no normalization)

            for row_idx, (pred_vec, gt_vec) in enumerate(zip(outputs, batch_y)):
                pred_vec_np = pred_vec.cpu().numpy()
                gt_vec_np = gt_vec.cpu().numpy()

                # Store per-class values for scatter plots
                for idx, tag in enumerate(probe_tags):
                    per_class_gt[tag].append(gt_vec_np[idx])
                    per_class_pred[tag].append(pred_vec_np[idx])

                # Extract subset for simplex visualization
                pred = pred_vec_np[pos_indices]
                gt = gt_vec_np[pos_indices]

                # Filter: need at least MIN_NONZERO_CLASSES nonzero GT classes
                if (gt > 0).sum() < MIN_NONZERO_CLASSES:
                    continue

                gt_sum = gt.sum()
                if gt_sum == 0:
                    continue
                if gt_sum < MIN_TOTAL_MASS:
                    continue

                # Ground truth: normalize to simplex
                gt_norm = gt / gt_sum

                # Prediction: use raw values WITHOUT normalization
                # This allows predictions to land outside the triangle
                pred_x, pred_y = barycentric_to_xy(*pred)
                gt_x, gt_y = barycentric_to_xy(*gt_norm)

                pred_points.append((pred_x, pred_y))
                gt_points.append((gt_x, gt_y))
                colors.append(gt_norm)
                raw_vectors.append(pred)
                activations.append(batch_x[row_idx].cpu().numpy())

    if not pred_points:
        print("No points met the filtering criteria.")
        return

    print(f"Collected {len(pred_points)} points for visualization")

    pred_xs, pred_ys = zip(*pred_points)
    gt_xs, gt_ys = zip(*gt_points)
    colors_arr = np.array(colors)

    # Map gt distributions to CMY for better blend visibility
    # POS1->C (0,1,1), POS2->M (1,0,1), POS3->Y (1,1,0)
    cmy_map = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cmy_colors = np.clip(colors_arr @ cmy_map, 0.0, 1.0)

    # Downsample for uniform density using triangular grid
    N_DIVISIONS = 30  # 30² = 900 equal triangular cells
    MAX_PER_CELL = 10  # Points to keep per cell

    # Convert to arrays for indexing
    pred_xs = np.array(pred_xs)
    pred_ys = np.array(pred_ys)
    gt_xs = np.array(gt_xs)
    gt_ys = np.array(gt_ys)
    activations_arr = np.array(activations)
    raw_vectors_arr = np.array(raw_vectors)

    # Compute triangular grid cell for each point
    # Scale barycentric coords and use (i, j) as cell key
    scaled = colors_arr * N_DIVISIONS
    cell_keys = [(int(s[0]), int(s[1])) for s in scaled.clip(0, N_DIVISIONS - 1e-6)]

    # Group indices by cell
    cells = defaultdict(list)
    for i, key in enumerate(cell_keys):
        cells[key].append(i)

    # Sample up to MAX_PER_CELL from each cell
    np.random.seed(42)
    sampled_indices = []
    for indices in cells.values():
        if len(indices) <= MAX_PER_CELL:
            sampled_indices.extend(indices)
        else:
            sampled_indices.extend(np.random.choice(indices, MAX_PER_CELL, replace=False))

    sampled_indices = np.array(sampled_indices)
    print(f"Downsampled from {len(pred_xs)} to {len(sampled_indices)} points ({len(cells)} cells)")

    # Apply downsampling to all arrays
    pred_xs = pred_xs[sampled_indices]
    pred_ys = pred_ys[sampled_indices]
    gt_xs = gt_xs[sampled_indices]
    gt_ys = gt_ys[sampled_indices]
    colors_arr = colors_arr[sampled_indices]
    cmy_colors = cmy_colors[sampled_indices]
    activations_arr = activations_arr[sampled_indices]
    raw_vectors_arr = raw_vectors_arr[sampled_indices]

    # Triangle vertices
    simplex_x = [0, 1, 0.5]
    simplex_y = [0, 0, math.sqrt(3) / 2]

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Predicted simplex (raw values, may be outside triangle)
    ax = axes[0]
    ax.triplot(simplex_x, simplex_y, "k-")
    ax.scatter(pred_xs, pred_ys, c=cmy_colors, s=20)
    ax.text(-0.05, -0.05, POS_CLASSES[0])
    ax.text(1.02, -0.05, POS_CLASSES[1])
    ax.text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    ax.set_title("Probe Predictions (Raw, Unnormalized)")
    ax.axis("off")

    # Right: Ground truth simplex
    ax = axes[1]
    ax.triplot(simplex_x, simplex_y, "k-")
    ax.scatter(gt_xs, gt_ys, c=cmy_colors, s=20)
    ax.text(-0.05, -0.05, POS_CLASSES[0])
    ax.text(1.02, -0.05, POS_CLASSES[1])
    ax.text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    ax.set_title("Ground Truth")
    ax.axis("off")

    plt.tight_layout()
    SIMPLEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SIMPLEX_FIG, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved simplex visualization to {SIMPLEX_FIG}")

    # Scatter plots: predicted vs ground truth per class
    # Y-axis is NOT constrained to [0, 1] since predictions can be outside
    cols = 4
    rows = math.ceil(len(probe_tags) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for idx, tag in enumerate(probe_tags):
        ax = axes[idx]
        gt_vals = np.array(per_class_gt[tag])
        pred_vals = np.array(per_class_pred[tag])
        ax.scatter(gt_vals, pred_vals, s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        ax.axhline(y=0, color='gray', linewidth=0.3, linestyle=':')
        ax.axhline(y=1, color='gray', linewidth=0.3, linestyle=':')
        ax.set_title(tag, fontsize=8)
        ax.set_xlim(0, 1)
        # Don't constrain y-axis to show predictions outside [0, 1]
        y_min = min(pred_vals.min(), 0) - 0.1
        y_max = max(pred_vals.max(), 1) + 0.1
        ax.set_ylim(y_min, y_max)

    for ax in axes[len(probe_tags):]:
        ax.axis('off')

    fig.suptitle("Predicted (y) vs Ground Truth (x) - Simple Linear Probe")
    fig.supxlabel("Ground Truth Probability")
    fig.supylabel("Raw Prediction (may be <0 or >1)")
    plt.tight_layout()
    SCATTER_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SCATTER_FIG, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved scatter plots to {SCATTER_FIG}")

    # Plotly 3D scatter: project activations onto probe direction basis
    # (activations_arr and raw_vectors_arr already downsampled above)
    colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in cmy_colors]

    weight_matrix = probe.linear.weight.detach().cpu().numpy()[pos_indices]  # (3, hidden_dim)
    U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
    basis = Vh  # (3, hidden_dim)

    proj_raw = activations_arr @ basis.T
    bias = probe.linear.bias.detach().cpu().numpy()[pos_indices]
    x0 = -np.linalg.pinv(weight_matrix) @ bias
    origin = basis @ x0
    axes_dirs = basis @ weight_matrix.T

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=proj_raw[:, 0],
            y=proj_raw[:, 1],
            z=proj_raw[:, 2],
            mode="markers",
            marker=dict(size=4, color=colors_hex, opacity=1.0),
        )
    )
    for i, tag in enumerate(POS_CLASSES):
        end = origin + axes_dirs[:, i]
        fig3d.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color="black", width=4),
                name=tag,
            )
        )
    fig3d.update_layout(
        title="Raw Probe Outputs (orthonormal projection)",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="white"),
            yaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="white"),
            zaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="white"),
            bgcolor="white",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig3d.write_html(PRED_3D_HTML)
    print(f"Saved 3D plot to {PRED_3D_HTML}")

    # Cosine similarity heatmap of all probe directions
    weight_matrix_full = probe.linear.weight.detach().cpu().numpy()
    norms = np.linalg.norm(weight_matrix_full, axis=1, keepdims=True) + 1e-8
    normed = weight_matrix_full / norms
    cosine = normed @ normed.T

    plt.figure(figsize=(8, 7))
    im = plt.imshow(cosine, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(probe_tags)), probe_tags, rotation=90, fontsize=6)
    plt.yticks(range(len(probe_tags)), probe_tags, fontsize=6)
    plt.title("Probe Direction Cosine Similarity (Simple Linear)")
    plt.tight_layout()
    HEATMAP_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HEATMAP_FIG, dpi=200)
    plt.close()
    print(f"Saved direction cosine heatmap to {HEATMAP_FIG}")

    # Print summary statistics (on downsampled data)
    print("\nPrediction statistics for selected POS classes (downsampled):")
    for i, pos in enumerate(POS_CLASSES):
        vals = raw_vectors_arr[:, i]
        print(f"  {pos}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, std={vals.std():.3f}")


if __name__ == "__main__":
    main()
