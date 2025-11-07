"""Activation extraction helpers for Gemma residual streams."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedModel
from tqdm.auto import tqdm

from .constants import DEFAULT_MODEL_NAME


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = torch.bfloat16,
) -> PreTrainedModel:
    """Load Gemma with hidden-state outputs enabled."""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model


def _select_hidden_state(hidden_states: Tuple[torch.Tensor, ...], layer_index: int) -> torch.Tensor:
    """Use 0-indexed transformer block coordinates."""

    if layer_index < 0 or layer_index >= len(hidden_states) - 1:
        raise ValueError(
            f"Layer index {layer_index} invalid for {len(hidden_states) - 1} transformer blocks."
        )
    return hidden_states[layer_index + 1]


def extract_activations(
    model: PreTrainedModel,
    dataloader: DataLoader,
    *,
    layer_index: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    desc: str = "Extracting activations",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model once over a dataloader and collect last-subtoken activations."""

    features = []
    labels = []

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_mask = batch["label_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        layer_hidden = _select_hidden_state(outputs.hidden_states, layer_index)

        selected = layer_hidden[label_mask]
        selected_labels = batch_labels[label_mask]

        if selected.shape[0] == 0:
            continue

        features.append(selected.to(dtype).cpu())
        labels.append(selected_labels.cpu())

    if not features:
        raise RuntimeError("No activations were collected from the dataloader.")

    activations = torch.cat(features, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return activations, label_tensor


def save_activation_cache(path: Path, activations: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"activations": activations, "labels": labels}, path)


def load_activation_cache(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path)
    return data["activations"], data["labels"]


def cache_path(base_dir: Path, model_name: str, layer_index: int, split: str) -> Path:
    safe_model = model_name.replace("/", "-")
    return base_dir / safe_model / f"layer_{layer_index}" / f"{split}.pt"
