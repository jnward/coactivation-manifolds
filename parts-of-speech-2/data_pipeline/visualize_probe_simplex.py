from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

# Configuration
CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
VAL_CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
PROBE_PATH = Path("models/linear_cone_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")  # tag order used to train the probe
PROBE_OUTPUT_IS_LOG = False
POS_CLASSES = ["ADV", "ADP", "SCONJ"]  # three classes to plot on simplex
# POS_CLASSES = ["CCONJ", "DET", "PRON"]
SIMPLEX_FIG = Path("models/simplex.png")
SCATTER_FIG = Path("models/pred_vs_gt.png")
PRED_3D_HTML = Path("models/pred_linear_3d.html")
HEATMAP_FIG = Path("models/probe_direction_cosine.png")
MIN_NONZERO_CLASSES = 2
MIN_TOTAL_MASS = 0.5
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
TRAIN_RATIO = 1.0  # if <1.0, split cached train into train/val subsets
SPLIT = "val"  # "train" or "val" when TRAIN_RATIO < 1
SEED = 1234
# Optional POS subset; set to None to use full probe tag set.
CLASS_SUBSET = None
PCA_DIM = 1024  # per-layer components in cache
TRUNCATE_PCA_DIM = None  # set to int < PCA_DIM to slice per-layer dims



def load_cache(cache_base: Path):
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


def internal_split(X: np.ndarray, Y: np.ndarray):
    if not (0 < TRAIN_RATIO < 1):
        return X, Y, None
    train_len = int(len(X) * TRAIN_RATIO)
    val_len = len(X) - train_len
    train_subset, val_subset = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
        [train_len, val_len],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_X = train_subset.dataset.tensors[0][train_subset.indices].numpy()
    train_Y = train_subset.dataset.tensors[1][train_subset.indices].numpy()
    val_X = val_subset.dataset.tensors[0][val_subset.indices].numpy()
    val_Y = val_subset.dataset.tensors[1][val_subset.indices].numpy()
    return train_X if SPLIT == "train" else val_X, train_Y if SPLIT == "train" else val_Y, (train_X, train_Y, val_X, val_Y)


def align_distributions(Y: np.ndarray, source_tags: List[str], target_tags: List[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(source_tags)}
    cols = [idx[t] for t in target_tags if t in idx]
    aligned = Y[:, cols]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned / row_sums


def truncate_features(X: np.ndarray) -> np.ndarray:
    if TRUNCATE_PCA_DIM is None or TRUNCATE_PCA_DIM >= PCA_DIM:
        return X
    if X.shape[1] % len(LAYERS) != 0:
        raise ValueError("X shape not divisible by number of layers; cannot truncate.")
    per_layer = X.shape[1] // len(LAYERS)
    if per_layer <= TRUNCATE_PCA_DIM:
        return X
    num_layers = len(LAYERS)
    return X.reshape(X.shape[0], num_layers, per_layer)[:, :, :TRUNCATE_PCA_DIM].reshape(X.shape[0], num_layers * TRUNCATE_PCA_DIM)


def apply_class_subset(X: np.ndarray, Y: np.ndarray, tag_names: List[str], subset: List[str]):
    missing = [t for t in subset if t not in tag_names]
    if missing:
        raise ValueError(f"Subset tags missing from tag_names: {missing}")
    idx = [tag_names.index(t) for t in subset]
    Y_sub = Y[:, idx]
    mask = (Y_sub.sum(axis=1) > 0) & ((Y_sub > 0).sum(axis=1) >= 2)
    Y_sub = Y_sub[mask]
    X_sub = X[mask]
    row_sums = np.clip(Y_sub.sum(axis=1, keepdims=True), 1e-8, None)
    Y_sub = Y_sub / row_sums
    dropped = len(Y) - len(Y_sub)
    return X_sub, Y_sub, subset, dropped


def load_probe(path: Path, hidden_dim: int, num_classes: int):
    state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model = LinearConeProbe(hidden_dim, num_classes)
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) else state)
    return model.eval().cuda()


class LinearConeProbe(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        cone_coords = torch.relu(self.linear(x))
        return cone_coords / (cone_coords.sum(dim=-1, keepdim=True) + 1e-8)


def barycentric_to_xy(p1: float, p2: float, p3: float) -> Tuple[float, float]:
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2)
    x = p1 * A[0] + p2 * B[0] + p3 * C[0]
    y = p1 * A[1] + p2 * B[1] + p3 * C[1]
    return x, y


def main():
    # Load caches
    X, Y, tag_names = load_cache(CACHE_BASE)
    X = truncate_features(X)
    # If you want val cache instead, point CACHE_BASE to it; VAL_CACHE_BASE is unused.
    if TRAIN_RATIO < 1.0:
        X, Y, _ = internal_split(X, Y)
        X = truncate_features(X)

    # Load probe tag order
    if TAG_NAMES_PATH.exists():
        import json
        probe_tags = json.load(TAG_NAMES_PATH.open())
    else:
        raise ValueError(f"Probe tag list not found at {TAG_NAMES_PATH}")
    # Align to probe tag order; require all tags to be present
    missing = [t for t in probe_tags if t not in tag_names]
    if missing:
        raise ValueError(f"Cache missing probe tags: {missing}")
    Y = align_distributions(Y, tag_names, probe_tags)
    tag_names = probe_tags

    # Optional class subset
    if CLASS_SUBSET:
        X, Y, subset_tags, dropped = apply_class_subset(X, Y, tag_names, CLASS_SUBSET)
        tag_names = subset_tags
        print(f"Applied class subset {CLASS_SUBSET}; dropped {dropped} rows.")

    pos_indices = [tag_names.index(pos) for pos in POS_CLASSES if pos in tag_names]
    if len(pos_indices) != 3:
        raise ValueError(f"POS_CLASSES {POS_CLASSES} not all found in tag_names {tag_names}")

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
    # Map gt distributions to CMY for better blend visibility
    colors_arr = np.array(colors)
    # POS1->C (0,1,1), POS2->M (1,0,1), POS3->Y (1,1,0)
    cmy_map = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    cmy_colors = np.clip(colors_arr @ cmy_map, 0.0, 1.0)  # shape (n,3)
    simplex_x = [0, 1, 0.5]
    simplex_y = [0, 0, math.sqrt(3) / 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Predicted simplex
    axes[0].triplot(simplex_x, simplex_y, "k-")
    axes[0].scatter(xs, ys, c=cmy_colors, s=20)
    axes[0].text(-0.05, -0.05, POS_CLASSES[0])
    axes[0].text(1.02, -0.05, POS_CLASSES[1])
    axes[0].text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    axes[0].set_title("Probe Predictions")
    axes[0].axis("off")

    # Ground-truth simplex
    gt_points = [barycentric_to_xy(*color) for color in colors_arr]
    gt_xs, gt_ys = zip(*gt_points)
    axes[1].triplot(simplex_x, simplex_y, "k-")
    axes[1].scatter(gt_xs, gt_ys, c=cmy_colors, s=20)
    axes[1].text(-0.05, -0.05, POS_CLASSES[0])
    axes[1].text(1.02, -0.05, POS_CLASSES[1])
    axes[1].text(0.5, math.sqrt(3) / 2 + 0.05, POS_CLASSES[2])
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    plt.tight_layout()
    SIMPLEX_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SIMPLEX_FIG, bbox_inches="tight", dpi=200)
    print(f"Saved simplex visualization to {SIMPLEX_FIG}")

    # Scatter plots
    cols = 4
    rows = math.ceil(len(tag_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, tag in enumerate(tag_names):
        ax = axes[idx]
        ax.scatter(per_class_gt[tag], per_class_pred[tag], s=5, alpha=0.5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_title(tag, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[len(tag_names) :]:
        ax.axis("off")
    fig.suptitle("Predicted vs Ground Truth Probabilities")
    plt.tight_layout()
    SCATTER_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SCATTER_FIG, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved scatter plots to {SCATTER_FIG}")

    # Plotly 3D scatter projected onto orthonormal basis of probe directions
    raw_arr = np.array(raw_vectors)
    activations_arr = np.array(activations)
    colors_rgb = np.clip(cmy_colors, 0.0, 1.0)
    colors_hex = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in colors_rgb]

    weight_matrix = probe.linear.weight.detach().cpu().numpy()[pos_indices]  # (3, hidden_dim)
    U, S, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
    basis = Vh  # (3, hidden_dim)

    proj_raw = activations_arr @ basis.T
    bias = probe.linear.bias.detach().cpu().numpy()[pos_indices]
    x0 = -np.linalg.pinv(weight_matrix) @ bias
    origin = basis @ x0
    axes_vecs = basis @ weight_matrix.T

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
        end = origin + axes_vecs[:, i]
        fig3d.add_trace(
            go.Scatter3d(
                x=[origin[0], end[0]],
                y=[origin[1], end[1]],
                z=[origin[2], end[2]],
                mode="lines",
                line=dict(color="white", width=4),
                name=tag,
            )
        )
    fig3d.update_layout(
        title="Raw Probe Outputs (orthonormal projection)",
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="black"),
            yaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="black"),
            zaxis=dict(showgrid=False, showticklabels=False, title=None, zeroline=False, backgroundcolor="black"),
            bgcolor="black",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
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
