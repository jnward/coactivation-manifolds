"""Shared helpers for training/evaluation."""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
import torch

NormalizationMode = Literal["softmax", "relu"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_logits(
    logits: torch.Tensor,
    mode: NormalizationMode = "softmax",
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mode == "softmax":
        return torch.softmax(logits, dim=dim)
    relu = torch.relu(logits)
    denom = relu.sum(dim=dim, keepdim=True) + eps
    return relu / denom


def accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    normalization: NormalizationMode = "softmax",
) -> float:
    probs = normalize_logits(logits, normalization)
    preds = torch.argmax(probs, dim=-1)
    correct = (preds == labels).sum().item()
    return correct / max(1, labels.shape[0])
