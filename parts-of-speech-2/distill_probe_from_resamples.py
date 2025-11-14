from __future__ import annotations

import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines as jsonlines

INPUT_PATH = "spacy_train_output_with_spacy_pos.jsonl"
MODEL_NAME = "google/gemma-2-2b"
TARGET_LAYER = 3
OUTPUT_PATH = Path("spacy_retagged/distilled_probe.pt")
TAG_NAMES_PATH = Path("spacy_retagged/distilled_probe_tag_names.json")
HTML_OUTPUT = Path("spacy_retagged/distilled_probe_val_examples.html")
TRAIN_RATIO = 0.9
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
MIN_COMPLETIONS = 10
SEED = 1234
HTML_SAMPLE_SIZE = 64


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


def build_distribution(completions: List[dict], tag_to_idx: dict[str, int]) -> np.ndarray | None:
    counts = np.zeros(len(tag_to_idx), dtype=np.float64)
    total = 0
    for comp in completions:
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


def collect_samples(records: List[dict], tag_names: List[str]) -> Tuple[List[Sample], List[dict]]:
    tag_to_idx = build_tag_map(tag_names)
    samples: List[Sample] = []
    infos: List[dict] = []
    for rec in records:
        candidate = rec.get("candidate", {})
        completions = rec.get("completions", [])
        dist = build_distribution(completions, tag_to_idx)
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
        hidden = outputs.hidden_states[TARGET_LAYER]
        for idx, s in enumerate(batch):
            features.append(hidden[idx, len(s.tokens) - 1].float().cpu().numpy())

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


def train_probe(X: np.ndarray, Y: np.ndarray):
    torch.manual_seed(SEED)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    train_len = int(len(dataset) * TRAIN_RATIO)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = DistillationProbe(X.shape[1], Y.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def kl_loss(log_probs, target_probs):
        target_safe = target_probs.clamp(min=1e-8)
        return torch.sum(target_safe * (torch.log(target_safe) - log_probs), dim=-1).mean()

    def r2_score(log_probs, target_probs):
        probs = log_probs.exp()
        numer = torch.sum((target_probs - probs) ** 2)
        denom = torch.sum((target_probs - target_probs.mean(dim=0, keepdim=True)) ** 2)
        if denom == 0:
            return torch.tensor(0.0, device=log_probs.device)
        return 1.0 - numer / denom

    best_val = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        train_r2_scores = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            log_probs = model(batch_x)
            loss = kl_loss(log_probs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_r2_scores.append(r2_score(log_probs.detach(), batch_y).item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_r2_scores = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                log_probs = model(batch_x)
                val_losses.append(kl_loss(log_probs, batch_y).item())
                val_r2_scores.append(r2_score(log_probs, batch_y).item())
        train_mean = float(np.mean(train_losses))
        train_r2 = float(np.mean(train_r2_scores))
        val_mean = float(np.mean(val_losses))
        val_r2 = float(np.mean(val_r2_scores))
        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"train KL: {train_mean:.4f} | train R²: {train_r2:.4f} | "
            f"val KL: {val_mean:.4f} | val R²: {val_r2:.4f}"
        )
        best_val = min(best_val, val_mean)

    return model, best_val, train_set, val_set


def format_top(dist: np.ndarray, tag_names: List[str], k: int = 5) -> str:
    idxs = np.argsort(dist)[::-1][:k]
    return ", ".join(f"{tag_names[i]} ({dist[i]*100:.1f}%)" for i in idxs)


def render_examples(
    model: DistillationProbe,
    loader: DataLoader,
    subset_indices: List[int],
    infos: List[dict],
    tag_names: List[str],
    html_path: Path,
    title: str,
):
    if not subset_indices:
        print(f"No {title} samples to render.")
        return
    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            probs = model(batch_x.cuda()).exp().cpu().numpy()
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


def main():
    tag_names, records = load_metadata_and_records(Path(INPUT_PATH))
    samples, infos = collect_samples(records, tag_names)
    if not samples:
        raise RuntimeError("No valid samples after filtering.")
    print(f"Collected {len(samples)} samples with >= {MIN_COMPLETIONS} completions.")

    X, Y = encode_hidden_states(samples)
    model, best_val, train_subset, val_subset = train_probe(X, Y)

    train_loader_vis = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader_vis = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    render_examples(
        model,
        train_loader_vis,
        train_subset.indices,
        infos,
        tag_names,
        HTML_OUTPUT.with_name("distilled_probe_old_train_examples.html"),
        "training",
    )
    render_examples(
        model,
        val_loader_vis,
        val_subset.indices,
        infos,
        tag_names,
        HTML_OUTPUT,
        "validation",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "tag_names": tag_names}, OUTPUT_PATH)
    TAG_NAMES_PATH.write_text(json.dumps(tag_names), encoding="utf-8")
    print(f"Saved distilled probe to {OUTPUT_PATH} (best val KL={best_val:.4f}).")


if __name__ == "__main__":
    main()
