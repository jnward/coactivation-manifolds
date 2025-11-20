from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import jsonlines as jsonl

import distill_probe_from_resamples as distill
from distill_probe_from_resamples import (
    DistillationProbe,
    LinearConeProbe,
    load_metadata_and_records,
    collect_samples,
    encode_hidden_states,
    SEED,
)


# Configuration constants
JSONL_PATH = Path("data/merged_subshards.jsonl")  # train JSONL
CACHE_PATH = Path("activation_cache/distill_cache_train.npz")  # train cache
PROBE_PATH = Path("models/linear_cone_probe_gemma2b.pt")
PROBE_CLASS = LinearConeProbe  # use DistillationProbe for softmax model
PROBE_OUTPUT_IS_LOG = False
POS_CLASSES = ["ADV", "ADP", "SCONJ"]
SIMPLEX_FIG = Path("models/simplex.png")
SCATTER_FIG = Path("models/pred_vs_gt.png")
PRED_3D_HTML = Path("models/pred_linear_3d.html")
HEATMAP_FIG = Path("models/probe_direction_cosine.png")
MIN_NONZERO_CLASSES = 2
MIN_TOTAL_MASS = 0.5
LAYERS = [2, 4, 6, 8]
# Internal split config
TRAIN_RATIO = 1.0  # if <1.0, split cached train into train/val subsets
SPLIT = "train"  # "train" or "val" subset to visualize when using internal split
# LAYERS = [0]

distill.LAYERS = LAYERS


def load_probe(path: Path, hidden_dim: int, num_classes: int):
    state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = PROBE_CLASS(hidden_dim, num_classes)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    return model.eval().cuda()


def barycentric_to_xy(p1: float, p2: float, p3: float) -> Tuple[float, float]:
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2)
    x = p1 * A[0] + p2 * B[0] + p3 * C[0]
    y = p1 * A[1] + p2 * B[1] + p3 * C[1]
    return x, y


def main():
    tag_names, records, _ = load_metadata_and_records(JSONL_PATH)
    if tag_names is None:
        pos_set = set()
        for rec in records:
            for comp in rec.get("completions", []):
                pos = comp.get("spacy_pos")
                if pos:
                    pos_set.add(pos)
        tag_names = sorted(pos_set)
        print(f"Inferred tag_names: {tag_names}")
    pos_indices = [tag_names.index(pos) for pos in POS_CLASSES if pos in tag_names]

    # Load cache or compute
    cache_valid = False
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH, allow_pickle=True)
        cached_layers = cache.get("layers")
        if cached_layers is not None and list(cached_layers) == LAYERS:
            X = cache["X"]
            Y = cache["Y"]
            cache_valid = True
    if not cache_valid:
        samples, _, _ = collect_samples(records, tag_names)
        X, Y = encode_hidden_states(samples)
        np.savez_compressed(CACHE_PATH, X=X, Y=Y, layers=np.array(LAYERS))

    # Optional internal split
    if TRAIN_RATIO < 1.0:
        train_len = int(len(X) * TRAIN_RATIO)
        val_len = len(X) - train_len
        train_subset, val_subset = torch.utils.data.random_split(
            torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
            [train_len, val_len],
            generator=torch.Generator().manual_seed(SEED),
        )
        if SPLIT == "train":
            X = train_subset.dataset.tensors[0][train_subset.indices].numpy()
            Y = train_subset.dataset.tensors[1][train_subset.indices].numpy()
        else:
            X = val_subset.dataset.tensors[0][val_subset.indices].numpy()
            Y = val_subset.dataset.tensors[1][val_subset.indices].numpy()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    probe = load_probe(PROBE_PATH, X.shape[1], len(tag_names))
    points = []
    colors = []
    raw_vectors = []
    activations = []
    per_class_gt = {tag: [] for tag in tag_names}
    per_class_pred = {tag: [] for tag in tag_names}

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            outputs = probe(batch_x)
            probs = outputs.exp() if PROBE_OUTPUT_IS_LOG else outputs
            raw_linear = probe.linear(batch_x)
            for row_idx, (pred_vec, gt_vec, raw_out) in enumerate(zip(probs, batch_y, raw_linear)):
                pred_vec_np = pred_vec.cpu().numpy()
                gt_vec_np = gt_vec.cpu().numpy()
                for idx, tag in enumerate(tag_names):
                    per_class_gt[tag].append(gt_vec_np[idx])
                    per_class_pred[tag].append(pred_vec_np[idx])
                pred = pred_vec_np[pos_indices]
                gt = gt_vec_np[pos_indices]
                if (gt > 0).sum() < MIN_NONZERO_CLASSES:
                    continue
                pred_sum = pred.sum()
                gt_sum = gt.sum()
                if pred_sum == 0 or gt_sum == 0:
                    continue
                if pred_sum < MIN_TOTAL_MASS and gt_sum < MIN_TOTAL_MASS:
                    continue
                pred_norm = pred / pred_sum
                gt_norm = gt / gt_sum
                x, y = barycentric_to_xy(*pred_norm)
                points.append((x, y))
                colors.append(gt_norm)
                raw_vectors.append(raw_out[pos_indices].cpu().numpy())
                activations.append(batch_x[row_idx].cpu().numpy())

    if not points:
        print("No points met the filtering criteria.")
        return

    xs, ys = zip(*points)
    colors_arr = np.array(colors)
    simplex_x = [0, 1, 0.5]
    simplex_y = [0, 0, math.sqrt(3) / 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Predicted simplex
    axes[0].triplot(simplex_x, simplex_y, 'k-')
    axes[0].scatter(xs, ys, c=colors_arr, s=20)
    axes[0].text(-0.05, -0.05, POS_CLASSES[0])
    axes[0].text(1.02, -0.05, POS_CLASSES[1])
    axes[0].text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    axes[0].set_title("Probe Predictions")
    axes[0].axis('off')

    # Ground-truth simplex
    gt_points = [barycentric_to_xy(*color) for color in colors_arr]
    gt_xs, gt_ys = zip(*gt_points)
    axes[1].triplot(simplex_x, simplex_y, 'k-')
    axes[1].scatter(gt_xs, gt_ys, c=colors_arr, s=20)
    axes[1].text(-0.05, -0.05, POS_CLASSES[0])
    axes[1].text(1.02, -0.05, POS_CLASSES[1])
    axes[1].text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    plt.tight_layout()
    SIMPLEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SIMPLEX_FIG, bbox_inches='tight', dpi=200)
    print(f"Saved simplex visualization to {SIMPLEX_FIG}")

    # Scatter plots
    cols = 4
    rows = math.ceil(len(tag_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, tag in enumerate(tag_names):
        ax = axes[idx]
        ax.scatter(per_class_gt[tag], per_class_pred[tag], s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        ax.set_title(tag, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[len(tag_names):]:
        ax.axis('off')
    fig.suptitle("Predicted vs Ground Truth Probabilities")
    plt.tight_layout()
    SCATTER_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SCATTER_FIG, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved scatter plots to {SCATTER_FIG}")

    # Plotly 3D scatter projected onto orthonormal basis of probe directions
    raw_arr = np.array(raw_vectors)
    activations_arr = np.array(activations)
    colors_rgb = (colors_arr * 255).astype(int)
    colors_hex = [f"rgb({r},{g},{b})" for r, g, b in colors_rgb]

    weight_matrix = probe.linear.weight.detach().cpu().numpy()[pos_indices]  # (3, hidden_dim)
    U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
    basis = Vh  # (3, hidden_dim)

    proj_raw = activations_arr @ basis.T
    bias = probe.linear.bias.detach().cpu().numpy()[pos_indices]
    x0 = -np.linalg.pinv(weight_matrix) @ bias
    origin = basis @ x0
    axes = basis @ weight_matrix.T

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=proj_raw[:, 0],
            y=proj_raw[:, 1],
            z=proj_raw[:, 2],
            mode="markers",
            marker=dict(size=4, color=colors_hex, opacity=0.7),
        )
    )
    for i, tag in enumerate(POS_CLASSES):
        end = origin + axes[:, i]
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
            xaxis_title=POS_CLASSES[0],
            yaxis_title=POS_CLASSES[1],
            zaxis_title=POS_CLASSES[2],
        ),
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
    plt.xticks(range(len(tag_names)), tag_names, rotation=90, fontsize=6)
    plt.yticks(range(len(tag_names)), tag_names, fontsize=6)
    plt.title("Probe Direction Cosine Similarity")
    plt.tight_layout()
    HEATMAP_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(HEATMAP_FIG, dpi=200)
    plt.close()
    print(f"Saved direction cosine heatmap to {HEATMAP_FIG}")


if __name__ == "__main__":
    main()
