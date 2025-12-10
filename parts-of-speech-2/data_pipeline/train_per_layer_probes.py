"""
Train individual probes for each layer in parallel.
Shows per-layer val R² after each epoch to compare layer effectiveness.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
CACHE_BASE = Path("activation_cache/distill_cache_train_pca")
VAL_CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
OUTPUT_DIR = Path("models/per_layer_probes")
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
PCA_DIM = 1024

BATCH_SIZE = 256
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 1234


class PerLayerProbes(nn.Module):
    """Multiple linear cone probes, one per layer."""

    def __init__(self, n_layers: int, input_dim: int, num_classes: int):
        super().__init__()
        self.n_layers = n_layers
        self.probes = nn.ModuleList([
            nn.Linear(input_dim, num_classes) for _ in range(n_layers)
        ])

    def forward(self, x):
        # x: (batch, n_layers, input_dim)
        # Returns: (n_layers, batch, num_classes)
        outputs = []
        for i, probe in enumerate(self.probes):
            layer_input = x[:, i, :]  # (batch, input_dim)
            cone = torch.relu(probe(layer_input))
            probs = cone / (cone.sum(dim=-1, keepdim=True) + 1e-8)
            outputs.append(probs)
        return torch.stack(outputs, dim=0)


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


def align_distributions(Y, source_tags, target_tags):
    """Align Y to target tag order."""
    idx = {t: i for i, t in enumerate(source_tags)}
    cols = [idx[t] for t in target_tags if t in idx]
    aligned = Y[:, cols]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned / row_sums


def l2_loss(probs, target_probs):
    return ((probs - target_probs) ** 2).sum(dim=-1).mean()


def r2_score(probs, target_probs):
    ss_res = torch.sum((target_probs - probs) ** 2)
    ss_tot = torch.sum((target_probs - target_probs.mean(dim=0, keepdim=True)) ** 2)
    if ss_tot == 0:
        return torch.tensor(0.0, device=probs.device)
    return 1.0 - ss_res / ss_tot


def evaluate_per_layer(model, val_loader, n_layers):
    """Compute R² for each layer's probe on validation set."""
    model.eval()
    # Accumulate predictions and targets
    all_probs = [[] for _ in range(n_layers)]
    all_targets = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.cuda()
            yb = yb.cuda()
            probs = model(xb)  # (n_layers, batch, num_classes)
            for i in range(n_layers):
                all_probs[i].append(probs[i])
            all_targets.append(yb)

    all_targets = torch.cat(all_targets, dim=0)
    r2_scores = []
    for i in range(n_layers):
        layer_probs = torch.cat(all_probs[i], dim=0)
        r2 = r2_score(layer_probs, all_targets).item()
        r2_scores.append(r2)

    return r2_scores


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load probe tag names (target order)
    with open(TAG_NAMES_PATH) as f:
        probe_tags = json.load(f)
    num_classes = len(probe_tags)

    # Load train data
    print("Loading train cache...")
    train_X, train_Y, train_tags = load_cache(CACHE_BASE)
    train_Y = align_distributions(train_Y, train_tags, probe_tags)

    # Load val data
    print("Loading val cache...")
    val_X, val_Y, val_tags = load_cache(VAL_CACHE_BASE)
    val_Y = align_distributions(val_Y, val_tags, probe_tags)

    # Reshape X to (n_samples, n_layers, pca_dim)
    n_layers = len(LAYERS)
    train_X = train_X.reshape(train_X.shape[0], n_layers, PCA_DIM)
    val_X = val_X.reshape(val_X.shape[0], n_layers, PCA_DIM)

    print(f"Train X shape: {train_X.shape}, Y shape: {train_Y.shape}")
    print(f"Val X shape: {val_X.shape}, Y shape: {val_Y.shape}")
    print(f"Layers: {LAYERS}")
    print(f"Classes: {num_classes} ({probe_tags})")

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)),
        batch_size=BATCH_SIZE
    )

    # Create model
    model = PerLayerProbes(n_layers, PCA_DIM, num_classes).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Track best R² for each layer
    best_r2 = [-float("inf")] * n_layers
    best_states = [None] * n_layers

    print(f"\nTraining {n_layers} probes in parallel for {EPOCHS} epochs...")
    print("=" * 100)

    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for xb, yb in pbar:
            xb = xb.cuda()  # (batch, n_layers, pca_dim)
            yb = yb.cuda()  # (batch, num_classes)

            optimizer.zero_grad()
            all_probs = model(xb)  # (n_layers, batch, num_classes)

            # Sum losses across all layers
            total_loss = torch.tensor(0.0, device="cuda")
            for layer_idx in range(n_layers):
                probs = all_probs[layer_idx]
                total_loss = total_loss + l2_loss(probs, yb)

            total_loss.backward()
            optimizer.step()

            epoch_losses.append(total_loss.item() / n_layers)
            pbar.set_postfix({"avg_loss": f"{np.mean(epoch_losses):.4f}"})

        # Validation: compute R² per layer
        val_r2_per_layer = evaluate_per_layer(model, val_loader, n_layers)

        # Update best states
        for i, r2 in enumerate(val_r2_per_layer):
            if r2 > best_r2[i]:
                best_r2[i] = r2
                best_states[i] = model.probes[i].state_dict().copy()

        # Print results
        r2_strs = [f"L{LAYERS[i]:2d}={r2:.3f}" for i, r2 in enumerate(val_r2_per_layer)]
        print(f"Epoch {epoch+1:2d} | " + " | ".join(r2_strs))

    print("=" * 100)
    print("\nFinal Results (Best Val R² per Layer):")
    print("-" * 50)
    sorted_layers = sorted(range(n_layers), key=lambda i: best_r2[i], reverse=True)
    for rank, i in enumerate(sorted_layers):
        print(f"  {rank+1:2d}. Layer {LAYERS[i]:2d}: R² = {best_r2[i]:.4f}")

    # Save best probes
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n_layers):
        probe_path = OUTPUT_DIR / f"probe_layer_{LAYERS[i]}.pt"
        torch.save({
            "state_dict": best_states[i],
            "layer": LAYERS[i],
            "input_dim": PCA_DIM,
            "num_classes": num_classes,
            "best_r2": best_r2[i],
        }, probe_path)
    print(f"\nSaved {n_layers} probes to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
