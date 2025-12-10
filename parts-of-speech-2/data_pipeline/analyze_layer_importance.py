# %% [markdown]
# # Layer Importance Analysis
# Visualize which layers are most important for each POS label by analyzing probe weights
# normalized by activation magnitudes.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configuration
PROBE_PATH = Path("models/linear_cone_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
CACHE_X_PATH = Path("activation_cache/distill_cache_val_pca_X.npy")
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCA_DIM = 1024  # per-layer

# %%
# Load probe
probe_state = torch.load(PROBE_PATH, map_location="cpu")
if isinstance(probe_state, dict) and "state_dict" in probe_state:
    weights = probe_state["state_dict"]["linear.weight"].numpy()  # (n_classes, hidden_dim)
    bias = probe_state["state_dict"]["linear.bias"].numpy()  # (n_classes,)
else:
    weights = probe_state["linear.weight"].numpy()
    bias = probe_state["linear.bias"].numpy()

# Load tag names
with open(TAG_NAMES_PATH) as f:
    tag_names = json.load(f)

print(f"Probe weights shape: {weights.shape}")
print(f"Number of classes: {len(tag_names)}")
print(f"Tags: {tag_names}")

# %%
# Load cached activations and labels
X = np.load(CACHE_X_PATH)
Y_raw = np.load(CACHE_X_PATH.with_name("distill_cache_val_pca_Y.npy"))
cache_meta = np.load(CACHE_X_PATH.with_name("distill_cache_val_pca_meta.npz"), allow_pickle=True)
cache_tags = list(cache_meta["tag_names"])

print(f"Activations shape: {X.shape}")
print(f"Labels shape (raw): {Y_raw.shape}")
print(f"Cache tags: {cache_tags}")
print(f"Probe tags: {tag_names}")

# Align Y to probe tag order
def align_distributions(Y, source_tags, target_tags):
    idx = {t: i for i, t in enumerate(source_tags)}
    cols = [idx[t] for t in target_tags if t in idx]
    aligned = Y[:, cols]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned / row_sums

Y = align_distributions(Y_raw, cache_tags, tag_names)
print(f"Labels shape (aligned): {Y.shape}")

n_samples = X.shape[0]
n_layers = len(LAYERS)
total_dim = X.shape[1]
per_layer_dim = total_dim // n_layers

print(f"Samples: {n_samples}, Layers: {n_layers}, Dim per layer: {per_layer_dim}")

# %%
# Compute layer-wise activation std
X_by_layer = X.reshape(n_samples, n_layers, per_layer_dim)
layer_stds = X_by_layer.std(axis=(0, 2))  # std across samples and features for each layer
print(f"Layer stds: {layer_stds}")

# %%
# Compute effective importance
# Reshape weights to (n_classes, n_layers, per_layer_dim)
n_classes = weights.shape[0]
W_by_layer = weights.reshape(n_classes, n_layers, per_layer_dim)

# L2 norm of weights per layer for each class
weight_norms = np.linalg.norm(W_by_layer, axis=2)  # (n_classes, n_layers)
print(f"Weight norms shape: {weight_norms.shape}")

# Effective importance = weight norm * activation std
effective_importance = weight_norms * layer_stds[None, :]  # (n_classes, n_layers)

# %%
# Plot heatmap - raw effective importance
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(effective_importance, aspect="auto", cmap="viridis")
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_yticks(range(n_classes))
ax.set_yticklabels(tag_names)
ax.set_xlabel("Layer")
ax.set_ylabel("POS Tag")
ax.set_title("Effective Layer Importance (weight norm × activation std)")
plt.colorbar(im, ax=ax, label="Importance")
plt.tight_layout()
plt.savefig("models/layer_importance_raw.png", dpi=150)
plt.show()

# %%
# Plot heatmap - normalized per class (shows relative layer importance within each class)
importance_normalized = effective_importance / effective_importance.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(importance_normalized, aspect="auto", cmap="viridis")
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_yticks(range(n_classes))
ax.set_yticklabels(tag_names)
ax.set_xlabel("Layer")
ax.set_ylabel("POS Tag")
ax.set_title("Relative Layer Importance (normalized per class)")
plt.colorbar(im, ax=ax, label="Fraction of importance")
plt.tight_layout()
plt.savefig("models/layer_importance_normalized.png", dpi=150)
plt.show()

# %%
# Summary: which layer is most important for each class?
most_important_layer = np.argmax(effective_importance, axis=1)
for i, tag in enumerate(tag_names):
    layer_idx = most_important_layer[i]
    print(f"{tag}: Layer {LAYERS[layer_idx]} (importance={effective_importance[i, layer_idx]:.4f})")

# %%
# Plot layer importance averaged across classes
avg_importance = effective_importance.mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(n_layers), avg_importance)
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_xlabel("Layer")
ax.set_ylabel("Average Effective Importance")
ax.set_title("Average Layer Importance Across All POS Tags")
plt.tight_layout()
plt.savefig("models/layer_importance_avg.png", dpi=150)
plt.show()

# %%
# Compute ablation-based importance heatmap (per class, per layer)
# For each layer, ablate it and measure per-class MSE increase

def compute_per_class_mse(X_input, Y_true, W, b):
    """Compute per-class MSE for linear cone probe predictions."""
    logits = X_input @ W.T + b
    cone = np.maximum(logits, 0)
    probs = cone / (cone.sum(axis=1, keepdims=True) + 1e-8)
    # Per-class MSE: average over samples
    per_class_mse = ((Y_true - probs) ** 2).mean(axis=0)  # (n_classes,)
    return per_class_mse

def ablate_single_layer(X_input, layer_idx, n_layers, per_layer_dim):
    """Zero out a single layer in X."""
    X_ablated = X_input.copy()
    start = layer_idx * per_layer_dim
    end = start + per_layer_dim
    X_ablated[:, start:end] = 0
    return X_ablated

# Baseline per-class MSE
baseline_per_class_mse = compute_per_class_mse(X, Y, weights, bias)
print(f"Baseline per-class MSE: {baseline_per_class_mse}")

# Compute ablation importance: MSE increase when each layer is removed
ablation_importance = np.zeros((n_classes, n_layers))
for layer_idx in range(n_layers):
    X_ablated = ablate_single_layer(X, layer_idx, n_layers, per_layer_dim)
    ablated_mse = compute_per_class_mse(X_ablated, Y, weights, bias)
    # Importance = how much MSE increases (higher = more important)
    ablation_importance[:, layer_idx] = ablated_mse - baseline_per_class_mse
    print(f"Layer {LAYERS[layer_idx]}: avg MSE increase = {ablation_importance[:, layer_idx].mean():.6f}")

# %%
# Plot ablation-based importance heatmap
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(ablation_importance, aspect="auto", cmap="viridis")
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_yticks(range(n_classes))
ax.set_yticklabels(tag_names)
ax.set_xlabel("Layer")
ax.set_ylabel("POS Tag")
ax.set_title("Ablation-Based Layer Importance (MSE increase when layer removed)")
plt.colorbar(im, ax=ax, label="MSE Increase")
plt.tight_layout()
plt.savefig("models/layer_importance_ablation.png", dpi=150)
plt.show()

# %%
# Plot ablation-based importance normalized per class
ablation_normalized = ablation_importance / (ablation_importance.sum(axis=1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(ablation_normalized, aspect="auto", cmap="viridis")
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_yticks(range(n_classes))
ax.set_yticklabels(tag_names)
ax.set_xlabel("Layer")
ax.set_ylabel("POS Tag")
ax.set_title("Ablation-Based Layer Importance (normalized per class)")
plt.colorbar(im, ax=ax, label="Fraction of importance")
plt.tight_layout()
plt.savefig("models/layer_importance_ablation_normalized.png", dpi=150)
plt.show()

# %%
# Average ablation importance per layer
avg_ablation_importance = ablation_importance.mean(axis=0)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(n_layers), avg_ablation_importance)
ax.set_xticks(range(n_layers))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_xlabel("Layer")
ax.set_ylabel("Average MSE Increase")
ax.set_title("Average Ablation Importance Across All POS Tags")
plt.tight_layout()
plt.savefig("models/layer_importance_ablation_avg.png", dpi=150)
plt.show()

# %%
# Iterative layer ablation analysis
# Zero-ablate each layer, measure R² drop, remove least impactful layer, repeat

def compute_r2(X_input, Y_true, W, b):
    """Compute R² score for linear cone probe predictions."""
    # Linear cone probe: ReLU then normalize
    logits = X_input @ W.T + b
    cone = np.maximum(logits, 0)
    probs = cone / (cone.sum(axis=1, keepdims=True) + 1e-8)

    # R² score
    ss_res = np.sum((Y_true - probs) ** 2)
    ss_tot = np.sum((Y_true - Y_true.mean(axis=0, keepdims=True)) ** 2)
    return 1 - ss_res / ss_tot

def ablate_layers(X_input, layer_indices_to_zero, n_layers, per_layer_dim):
    """Zero out specified layer indices in X."""
    X_ablated = X_input.copy()
    for layer_idx in layer_indices_to_zero:
        start = layer_idx * per_layer_dim
        end = start + per_layer_dim
        X_ablated[:, start:end] = 0
    return X_ablated

# Baseline R² with all layers
baseline_r2 = compute_r2(X, Y, weights, bias)
print(f"Baseline R² (all layers): {baseline_r2:.4f}")

# Iterative ablation
remaining_layers = list(range(n_layers))  # indices into LAYERS
ablation_history = []  # (removed_layer, r2_after_removal, layers_remaining)
ablated_layer_indices = []  # layer indices that have been removed

current_r2 = baseline_r2
ablation_history.append((None, current_r2, len(remaining_layers)))

while len(remaining_layers) > 0:
    # For each remaining layer, compute R² if we ablate it
    r2_if_ablated = {}
    for layer_idx in remaining_layers:
        test_ablated = ablated_layer_indices + [layer_idx]
        X_test = ablate_layers(X, test_ablated, n_layers, per_layer_dim)
        r2 = compute_r2(X_test, Y, weights, bias)
        r2_if_ablated[layer_idx] = r2

    # Find layer with smallest R² drop (least impactful to remove)
    # i.e., highest R² after ablation
    least_impactful = max(r2_if_ablated, key=r2_if_ablated.get)
    new_r2 = r2_if_ablated[least_impactful]

    # Remove it
    remaining_layers.remove(least_impactful)
    ablated_layer_indices.append(least_impactful)

    ablation_history.append((LAYERS[least_impactful], new_r2, len(remaining_layers)))
    print(f"Removed layer {LAYERS[least_impactful]:2d} -> R²={new_r2:.4f} ({len(remaining_layers)} layers remaining)")

    current_r2 = new_r2

# %%
# Plot ablation results
removed_layers = [h[0] for h in ablation_history]
r2_values = [h[1] for h in ablation_history]
n_remaining = [h[2] for h in ablation_history]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(r2_values)), r2_values, 'b-o', linewidth=2, markersize=8)
ax.set_xticks(range(len(r2_values)))
xlabels = ["All"] + [str(l) for l in removed_layers[1:]]
ax.set_xticklabels(xlabels, rotation=45, ha="right")
ax.set_xlabel("Layer Removed (cumulative)")
ax.set_ylabel("R² Score")
ax.set_title("Iterative Layer Ablation: Removing Least Impactful Layer Each Step")
ax.axhline(y=baseline_r2, color='r', linestyle='--', alpha=0.5, label=f"Baseline R²={baseline_r2:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations for R² values
for i, (r2, label) in enumerate(zip(r2_values, xlabels)):
    ax.annotate(f"{r2:.3f}", (i, r2), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("models/layer_ablation_curve.png", dpi=150)
plt.show()

# %%
# Print removal order (most to least important, reversed)
print("\nLayer importance ranking (most important = removed last):")
for i, layer in enumerate(reversed(removed_layers[1:])):
    print(f"  {i+1}. Layer {layer}")


# %%