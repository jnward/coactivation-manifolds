"""Linear probe definition plus training & evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .constants import POS_TAGS
from .utils import NormalizationMode, normalize_logits


class LinearProbe(nn.Module):
    def __init__(self, d_model: int, num_labels: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc_softmax: float
    val_acc_relu: float


def _prediction_stats(
    probe: LinearProbe,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
    normalization: NormalizationMode = "softmax",
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Return (correct, total, per_class_correct, per_class_total)."""

    num_labels = probe.linear.out_features
    per_class_correct = torch.zeros(num_labels, dtype=torch.long)
    per_class_total = torch.zeros(num_labels, dtype=torch.long)
    total_correct = 0
    total = 0

    probe.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            logits = probe(features)
            probs = normalize_logits(logits, normalization)
            preds = probs.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total += labels.shape[0]

            labels_cpu = labels.to("cpu")
            preds_cpu = preds.to("cpu")

            per_class_total += torch.bincount(labels_cpu, minlength=num_labels)
            mask = preds_cpu == labels_cpu
            if mask.any():
                per_class_correct += torch.bincount(
                    labels_cpu[mask], minlength=num_labels
                )

    return total_correct, total, per_class_correct, per_class_total


def average_loss(
    probe: LinearProbe,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
) -> float:
    """Compute mean cross-entropy loss over a dataloader."""

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_examples = 0

    probe.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            logits = probe(features)
            loss = criterion(logits, labels)
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(1, total_examples)


def evaluate(
    probe: LinearProbe,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
    normalization: NormalizationMode = "softmax",
) -> float:
    total_correct, total, _, _ = _prediction_stats(
        probe,
        dataloader,
        device=device,
        normalization=normalization,
    )
    return total_correct / max(1, total)


def per_tag_recall(
    probe: LinearProbe,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
    normalization: NormalizationMode = "softmax",
    tag_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float | None]:
    _, _, per_class_correct, per_class_total = _prediction_stats(
        probe,
        dataloader,
        device=device,
        normalization=normalization,
    )
    if tag_names is None:
        tag_names = list(POS_TAGS[: probe.linear.out_features])
    recalls: dict[str, float | None] = {}
    for idx, tag in enumerate(tag_names):
        total = per_class_total[idx].item()
        if total == 0:
            recalls[tag] = None
        else:
            recalls[tag] = per_class_correct[idx].item() / total
    return recalls


def confusion_matrix_counts(
    probe: LinearProbe,
    dataloader: DataLoader,
    *,
    device: str = "cuda",
    normalization: NormalizationMode = "softmax",
) -> np.ndarray:
    """Return a numpy confusion matrix (rows=true tags, cols=predicted tags)."""

    num_labels = probe.linear.out_features
    matrix = np.zeros((num_labels, num_labels), dtype=np.int64)

    probe.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            logits = probe(features)
            probs = normalize_logits(logits, normalization)
            preds = probs.argmax(dim=-1)

            labels_np = labels.to("cpu").numpy()
            preds_np = preds.to("cpu").numpy()
            np.add.at(matrix, (labels_np, preds_np), 1)

    return matrix


def bake_input_normalization(
    probe: LinearProbe,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """Fold (x-mean)/std normalization into the probe weights."""

    if mean is None or std is None:
        return
    mean = mean.to(probe.linear.weight.device, dtype=probe.linear.weight.dtype)
    std = std.to(probe.linear.weight.device, dtype=probe.linear.weight.dtype)
    with torch.no_grad():
        weight = probe.linear.weight
        bias = probe.linear.bias
        weight.div_(std)
        bias.sub_((weight * mean).sum(dim=1))


def train_probe(
    probe: LinearProbe,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = "cuda",
) -> list[TrainStats]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    probe.to(device)
    history: list[TrainStats] = []

    for epoch in range(1, epochs + 1):
        probe.train()
        running_loss = 0.0
        steps = 0
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch} training", leave=False):
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

        val_softmax = evaluate(probe, val_loader, device=device, normalization="softmax")
        val_relu = evaluate(probe, val_loader, device=device, normalization="relu")
        val_loss = average_loss(probe, val_loader, device=device)
        history.append(
            TrainStats(
                epoch=epoch,
                train_loss=running_loss / max(1, steps),
                val_loss=val_loss,
                val_acc_softmax=val_softmax,
                val_acc_relu=val_relu,
            )
        )

    return history


def save_probe(probe: LinearProbe, path: Path, *, metadata: Dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": probe.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_probe(path: Path, *, map_location: str | torch.device = "cpu") -> Tuple[LinearProbe, Dict]:
    payload = torch.load(path, map_location=map_location)
    metadata = payload.get("metadata", {})
    hidden_size = metadata.get("hidden_size")
    num_labels = metadata.get("num_labels")
    if hidden_size is None or num_labels is None:
        raise ValueError("Probe metadata missing hidden_size/num_labels; cannot rebuild model.")
    probe = LinearProbe(hidden_size, num_labels)
    probe.load_state_dict(payload["state_dict"])
    return probe, metadata
