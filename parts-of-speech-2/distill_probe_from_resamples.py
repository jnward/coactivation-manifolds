from __future__ import annotations

import argparse
import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines as jsonlines

INPUT_PATH = "filtered_train_resamples_with_spacy_pos.jsonl"
VAL_INPUT_PATH = "filtered_val_resamples_with_spacy_pos.jsonl"
MODEL_NAME = "google/gemma-2-2b"
LAYERS = [2, 4, 6, 8]
OUTPUT_PATH = Path("spacy_retagged/distilled_probe_gemma2b.pt")
LINEAR_CONE_OUTPUT_PATH = Path("spacy_retagged/linear_cone_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("spacy_retagged/distilled_probe_tag_names.json")
HTML_OUTPUT = Path("spacy_retagged/distilled_probe_val_examples.html")
LINEAR_CONE_HTML_OUTPUT = Path("spacy_retagged/linear_cone_probe_val_examples.html")
CACHE_PATH = Path("spacy_retagged/distill_cache_train.npz")
VAL_CACHE_PATH = Path("spacy_retagged/distill_cache_val.npz")
TRAIN_RATIO = 1.0
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-2
MIN_COMPLETIONS = 1
SEED = 1234
HTML_SAMPLE_SIZE = 64
SAVE_EPOCH = "last"  # "best" or "last"
NUM_TRAIN_RESAMPLES = 100 # completions to use per train example (None for all)
NUM_TRAIN_EXAMPLES = int(45374 * 1)  # limit number of train examples (None for all)


@dataclass
class Sample:
    tokens: List[int]
    distribution: np.ndarray


def load_metadata_and_records(path: Path) -> Tuple[List[str], List[dict]]:
    tag_names = None
    records = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            if isinstance(obj, dict) and "metadata" in obj:
                meta = obj["metadata"]
                tag_names = meta.get("tag_names")
                continue
            records.append(obj)
    if tag_names is None:
        raise ValueError("Missing tag_names in metadata.")
    return tag_names, records


def build_tag_map(tag_names: List[str]) -> dict[str, int]:
    return {tag: idx for idx, tag in enumerate(tag_names)}


def build_distribution(completions: List[dict], tag_to_idx: dict[str, int], max_items: int | None = None) -> np.ndarray | None:
    counts = np.zeros(len(tag_to_idx), dtype=np.float64)
    total = 0
    for comp in completions[: max_items or len(completions)]:
        label = comp.get("spacy_pos")
        if not label:
            continue
        idx = tag_to_idx.get(label)
        if idx is None:
            continue
        counts[idx] += 1
        total += 1
    if total < MIN_COMPLETIONS:
        return None
    counts /= total
    return counts


def collect_samples(records: List[dict], tag_names: List[str], max_completions: int | None = None) -> Tuple[List[Sample], List[dict]]:
    tag_to_idx = build_tag_map(tag_names)
    samples: List[Sample] = []
    infos: List[dict] = []
    for rec in records:
        candidate = rec.get("candidate", {})
        completions = rec.get("completions", [])
        dist = build_distribution(completions, tag_to_idx, max_items=max_completions)
        if dist is None:
            continue
        tokens = candidate.get("prefix_token_ids")
        if not tokens:
            continue
        samples.append(Sample(tokens=tokens, distribution=dist))
        infos.append(
            {
                "sentence": candidate.get("sentence_text", ""),
                "prefix": candidate.get("prefix_text", ""),
                "target": candidate.get("target_token_text", ""),
                "dataset_idx": candidate.get("dataset_idx", -1),
            }
        )
    return samples, infos


def encode_hidden_states(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for Gemma encoding.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    features = []
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Encoding prefixes"):
        batch = samples[i : i + BATCH_SIZE]
        max_len = max(len(s.tokens) for s in batch)
        input_ids = []
        attention_mask = []
        for s in batch:
            padded = s.tokens + [tokenizer.pad_token_id] * (max_len - len(s.tokens))
            mask = [1] * len(s.tokens) + [0] * (max_len - len(s.tokens))
            input_ids.append(padded)
            attention_mask.append(mask)
        input_ids = torch.tensor(input_ids, device="cuda")
        attention_mask = torch.tensor(attention_mask, device="cuda")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states
        selected_layers = [hidden_states[layer] for layer in LAYERS]
        for idx, s in enumerate(batch):
            concat_feat = []
            pos = len(s.tokens) - 1
            for layer_tensor in selected_layers:
                concat_feat.append(layer_tensor[idx, pos].float())
            features.append(torch.cat(concat_feat, dim=-1).cpu().numpy())

    distributions = np.stack([s.distribution for s in samples]).astype(np.float32)
    X = np.stack(features).astype(np.float32)
    return X, distributions


class DistillationProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return torch.log_softmax(logits, dim=-1)


class LinearConeProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        cone_coords = torch.relu(self.linear(x))  # Project to positive cone
        return cone_coords / (cone_coords.sum(dim=-1, keepdim=True) + 1e-8)


def train_probe_with_val(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray):
    torch.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)), batch_size=BATCH_SIZE)

    # Train softmax probe with KL divergence (optional; currently disabled)
    softmax_model = None
    best_softmax_val = None
    best_softmax_state = None

    def kl_loss(log_probs, target_probs):
        target_safe = target_probs.clamp(min=1e-8)
        return torch.sum(target_safe * (torch.log(target_safe) - log_probs), dim=-1).mean()

    def l2_loss(probs, target_probs):
        return ((probs - target_probs) ** 2).sum(dim=-1).mean()

    def r2_score(probs, target_probs):
        numer = torch.sum((target_probs - probs) ** 2)
        denom = torch.sum((target_probs - target_probs.mean(dim=0, keepdim=True)) ** 2)
        if denom == 0:
            return torch.tensor(0.0, device=probs.device)
        return 1.0 - numer / denom

    def tv_distance(probs, target_probs):
        return 0.5 * torch.sum(torch.abs(probs - target_probs), dim=-1).mean()

    def tv_distance(probs, target_probs):
        return 0.5 * torch.sum(torch.abs(probs - target_probs), dim=-1).mean()

    # best_softmax_val = float("inf")
    # best_softmax_state = copy.deepcopy(softmax_model.state_dict())
    # for epoch in range(EPOCHS):
    #     softmax_model.train()
    #     train_losses = []
    #     train_r2_scores = []
    #     train_tv_scores = []
    #     for batch_x, batch_y in train_loader:
    #         batch_x = batch_x.cuda()
    #         batch_y = batch_y.cuda()
    #         softmax_optimizer.zero_grad()
    #         log_probs = softmax_model(batch_x)
    #         loss = kl_loss(log_probs, batch_y)
    #         loss.backward()
    #         softmax_optimizer.step()
    #         train_losses.append(loss.item())
    #         probs = log_probs.exp().detach()
    #         train_r2_scores.append(r2_score(probs, batch_y).item())
    #         train_tv_scores.append(tv_distance(probs, batch_y).item())

    #     softmax_model.eval()
    #     with torch.no_grad():
    #         val_losses = []
    #         val_r2_scores = []
    #         val_tv_scores = []
    #         for batch_x, batch_y in val_loader:
    #             batch_x = batch_x.cuda()
    #             batch_y = batch_y.cuda()
    #             log_probs = softmax_model(batch_x)
    #             probs = log_probs.exp()
    #             val_losses.append(kl_loss(log_probs, batch_y).item())
    #             val_r2_scores.append(r2_score(probs, batch_y).item())
    #             val_tv_scores.append(tv_distance(probs, batch_y).item())
    #     train_mean = float(np.mean(train_losses))
    #     train_r2 = float(np.mean(train_r2_scores))
    #     train_tv = float(np.mean(train_tv_scores))
    #     val_mean = float(np.mean(val_losses))
    #     val_r2 = float(np.mean(val_r2_scores))
    #     val_tv = float(np.mean(val_tv_scores))
    #     print(
    #         f"Epoch {epoch+1}/{EPOCHS} - "
    #         f"train KL: {train_mean:.4f} | train R²: {train_r2:.4f} | train TV: {train_tv:.4f} | "
    #         f"val KL: {val_mean:.4f} | val R²: {val_r2:.4f} | val TV: {val_tv:.4f}"
    #     )
    #     if val_mean < best_softmax_val:
    #         best_softmax_val = val_mean
    #         best_softmax_state = copy.deepcopy(softmax_model.state_dict())

    # Train linear cone probe with L2 loss
    print("\n=== Training Linear Cone Probe (L2 Loss) ===")
    torch.manual_seed(SEED)  # Reset for fair comparison
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)), batch_size=BATCH_SIZE)
    
    cone_model = LinearConeProbe(X.shape[1], Y.shape[1]).cuda()
    cone_optimizer = torch.optim.AdamW(cone_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_cone_val = float("inf")
    best_cone_state = copy.deepcopy(cone_model.state_dict())
    for epoch in range(EPOCHS):
        cone_model.train()
        train_losses = []
        train_r2_scores = []
        train_tv_scores = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            cone_optimizer.zero_grad()
            probs = cone_model(batch_x)
            loss = l2_loss(probs, batch_y)
            loss.backward()
            cone_optimizer.step()
            train_losses.append(loss.item())
            train_r2_scores.append(r2_score(probs.detach(), batch_y).item())
            train_tv_scores.append(tv_distance(probs.detach(), batch_y).item())

        cone_model.eval()
        with torch.no_grad():
            val_losses = []
            val_r2_scores = []
            val_tv_scores = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                probs = cone_model(batch_x)
                val_losses.append(l2_loss(probs, batch_y).item())
                val_r2_scores.append(r2_score(probs, batch_y).item())
                val_tv_scores.append(tv_distance(probs, batch_y).item())
        train_mean = float(np.mean(train_losses))
        train_r2 = float(np.mean(train_r2_scores))
        train_tv = float(np.mean(train_tv_scores))
        val_mean = float(np.mean(val_losses))
        val_r2 = float(np.mean(val_r2_scores))
        val_tv = float(np.mean(val_tv_scores))
        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"train L2: {train_mean:.4f} | train R²: {train_r2:.4f} | train TV: {train_tv:.4f} | "
            f"val L2: {val_mean:.4f} | val R²: {val_r2:.4f} | val TV: {val_tv:.4f}"
        )
        if val_mean < best_cone_val:
            best_cone_val = val_mean
            best_cone_state = copy.deepcopy(cone_model.state_dict())

    print(f"\n=== Final Results ===")
    # print(f"Softmax Probe - Best val KL: {best_softmax_val:.4f}")
    print(f"Linear Cone Probe - Best val L2: {best_cone_val:.4f}")

    return (
        softmax_model,
        cone_model,
        best_softmax_val,
        best_cone_val,
        best_softmax_state,
        best_cone_state,
    )


def format_top(dist: np.ndarray, tag_names: List[str], k: int = 5) -> str:
    idxs = np.argsort(dist)[::-1][:k]
    return ", ".join(f"{tag_names[i]} ({dist[i]*100:.1f}%)" for i in idxs)


def render_examples(
    model: nn.Module,
    loader: DataLoader,
    subset_indices: List[int],
    infos: List[dict],
    tag_names: List[str],
    html_path: Path,
    title: str,
    is_log_space: bool = True,
):
    if model is None:
        return
    if not subset_indices:
        print(f"No {title} samples to render.")
        return
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            out = model(batch_x.cuda())
            if is_log_space:
                probs = out.exp().cpu().numpy()
            else:
                probs = out.cpu().numpy()
            preds.append(probs)
            truths.append(batch_y.numpy())
    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)

    selected_infos = [infos[i] for i in subset_indices]
    rng = np.random.default_rng(SEED)
    sample_count = min(len(selected_infos), HTML_SAMPLE_SIZE)
    select_idx = rng.choice(len(selected_infos), size=sample_count, replace=False)

    rows = []
    for idx in select_idx:
        info = selected_infos[idx]
        pred = preds[idx]
        truth = truths[idx]
        rows.append(
            "<tr>"
            f"<td>{info.get('dataset_idx', -1)}</td>"
            f"<td>{html.escape(info.get('target', ''))}</td>"
            f"<td>{html.escape(format_top(pred, tag_names))}</td>"
            f"<td>{html.escape(format_top(truth, tag_names))}</td>"
            f"<td>{html.escape(info.get('prefix', ''))}</td>"
            f"<td>{html.escape(info.get('sentence', ''))}</td>"
            "</tr>"
        )

    html_body = "\n".join(rows)
    html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Distilled Probe Samples - {title}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Random {title} samples ({sample_count})</h1>
<table>
<thead>
<tr>
<th>Dataset idx</th>
<th>Target token</th>
<th>Probe top probs</th>
<th>Ground truth top probs</th>
<th>Prefix (incl. target)</th>
<th>Sentence</th>
</tr>
</thead>
<tbody>
{html_body}
</tbody>
</table>
</body>
</html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {title} sample HTML to {html_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distill probe from resampled POS distributions")
    parser.add_argument("--input", type=Path, default=Path(INPUT_PATH))
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--recompute-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    cache_path = args.cache

    tag_names, records = load_metadata_and_records(input_path)
    samples, infos = collect_samples(records, tag_names, max_completions=NUM_TRAIN_RESAMPLES)
    if not samples:
        raise RuntimeError("No valid samples after filtering.")
    print(f"Collected {len(samples)} samples with >= {MIN_COMPLETIONS} completions.")

    use_cache = not args.no_cache
    cache_valid = False
    if use_cache and cache_path.exists() and not args.recompute_cache:
        cache = np.load(cache_path, allow_pickle=True)
        cached_layers = cache.get("layers")
        if cached_layers is not None and list(cached_layers) == LAYERS:
            X = cache["X"]
            cache_valid = True
            print(f"Loaded cached activations from {cache_path}")
        else:
            print("Cache layer configuration mismatch; recomputing activations.")
    if not cache_valid:
        X, _ = encode_hidden_states(samples)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, X=X, layers=np.array(LAYERS))
            print(f"Cached activations to {cache_path}")
    Y = np.array([s.distribution for s in samples], dtype=np.float32)
    if NUM_TRAIN_EXAMPLES is not None:
        X = X[:NUM_TRAIN_EXAMPLES]
        Y = Y[:NUM_TRAIN_EXAMPLES]

    # Load validation set (no caching, since size is smaller)
    val_tag_names, val_records = load_metadata_and_records(Path(VAL_INPUT_PATH))
    if val_tag_names != tag_names:
        raise ValueError("Tag names for train/val do not match.")
    val_samples, val_infos = collect_samples(val_records, tag_names, max_completions=None)
    if not val_samples:
        raise RuntimeError("No validation samples after filtering.")
    print(f"Collected {len(val_samples)} validation samples with >= {MIN_COMPLETIONS} completions.")

    val_cache_valid = False
    if use_cache and VAL_CACHE_PATH.exists() and not args.recompute_cache:
        val_cache = np.load(VAL_CACHE_PATH, allow_pickle=True)
        cached_layers = val_cache.get("layers")
        if cached_layers is not None and list(cached_layers) == LAYERS:
            val_X = val_cache["X"]
            val_cache_valid = True
            print(f"Loaded cached val activations from {VAL_CACHE_PATH}")
        else:
            print("Val cache layer mismatch; recomputing activations.")
    if not val_cache_valid:
        val_X, _ = encode_hidden_states(val_samples)
        if use_cache:
            VAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(VAL_CACHE_PATH, X=val_X, layers=np.array(LAYERS))
            print(f"Cached val activations to {VAL_CACHE_PATH}")
    val_Y = np.array([s.distribution for s in val_samples], dtype=np.float32)

    (
        softmax_model,
        cone_model,
        best_softmax_val,
        best_cone_val,
        best_softmax_state,
        best_cone_state,
    ) = train_probe_with_val(X, Y, val_X, val_Y)

    train_loader_vis = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=BATCH_SIZE, shuffle=False)
    val_loader_vis = DataLoader(TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y)), batch_size=BATCH_SIZE, shuffle=False)

    # Render softmax probe examples
    render_examples(
        softmax_model,
        train_loader_vis,
        list(range(len(samples))),
        infos,
        tag_names,
        HTML_OUTPUT.with_name("distilled_probe_train_examples.html"),
        "Softmax Probe - training",
        is_log_space=True,
    )
    render_examples(
        softmax_model,
        val_loader_vis,
        list(range(len(val_samples))),
        val_infos,
        tag_names,
        HTML_OUTPUT,
        "Softmax Probe - validation",
        is_log_space=True,
    )

    # Render linear cone probe examples
    render_examples(
        cone_model,
        train_loader_vis,
        list(range(len(samples))),
        infos,
        tag_names,
        LINEAR_CONE_HTML_OUTPUT.with_name("linear_cone_probe_train_examples.html"),
        "Linear Cone Probe - training",
        is_log_space=False,
    )
    render_examples(
        cone_model,
        val_loader_vis,
        list(range(len(val_samples))),
        val_infos,
        tag_names,
        LINEAR_CONE_HTML_OUTPUT,
        "Linear Cone Probe - validation",
        is_log_space=False,
    )

    # Save both models (best or last)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mode = SAVE_EPOCH.lower()
    if mode == "best":
        softmax_state = best_softmax_state
        cone_state = best_cone_state
    elif mode == "last":
        softmax_state = None if softmax_model is None else softmax_model.state_dict()
        cone_state = cone_model.state_dict()
    else:
        raise ValueError("SAVE_EPOCH must be 'best' or 'last'")

    if softmax_state is not None:
        torch.save({"state_dict": softmax_state, "tag_names": tag_names}, OUTPUT_PATH)
    torch.save({"state_dict": cone_state, "tag_names": tag_names}, LINEAR_CONE_OUTPUT_PATH)
    TAG_NAMES_PATH.write_text(json.dumps(tag_names), encoding="utf-8")

    if softmax_state is not None:
        print(
            f"\nSaved softmax probe to {OUTPUT_PATH} (best val KL={best_softmax_val:.4f}, mode={mode})."
        )
    else:
        print("Softmax probe skipped (not saved).")
    print(
        f"Saved linear cone probe to {LINEAR_CONE_OUTPUT_PATH} (best val L2={best_cone_val:.4f}, mode={mode})."
    )


if __name__ == "__main__":
    main()
