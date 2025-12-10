from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import copy
import itertools

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines as jsonlines

DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "merged_shards.jsonl"
VAL_INPUT_PATH = DATA_DIR / "../../filtered_train_resamples_with_spacy_pos.jsonl"
MODEL_NAME = "google/gemma-2-2b"
LAYERS = [2, 4, 6, 8]
LINEAR_CONE_OUTPUT_PATH = Path("models/linear_cone_probe_gemma2b_sweep.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
CACHE_BASE = Path("activation_cache/distill_cache_train")
VAL_CACHE_BASE = Path("activation_cache/distill_cache_val")
FORCE_REGEN_CACHE = False
TRAIN_RATIO = 1.0
NUM_TRAIN_RESAMPLES = None  # completions to use per train example (None for all)
NUM_TRAIN_EXAMPLES = None  # limit number of train examples (None for all)
FILTER_SINGLE_POS = True
SEED = 1234
SAVE_EPOCH = "last"  # "best" or "last"
WARMUP_STEPS = 1000
WEIGHT_DECAY = 1e-3

# Sweep grids (lists). Every combo will be trained with identical data ordering.
SWEEP_BATCH_SIZES = [1024]
SWEEP_LRS = [1e-3]
SWEEP_EPOCHS = [25]

# Optional class subset. If set, keep only these tags, drop rows with <2 nonzero
# entries among them, and renormalize Y to the subset.
# CLASS_SUBSET = ["ADV", "ADP", "SCONJ"]  # e.g., ["NOUN", "PROPN", "ADJ"]
CLASS_SUBSET = None


@dataclass
class Sample:
    tokens: List[int]
    distribution: np.ndarray


def load_metadata_and_records(path: Path) -> Tuple[List[str] | None, List[dict], List[str]]:
    tag_names = None
    records = []
    pos_list = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            if isinstance(obj, dict) and "metadata" in obj:
                meta = obj["metadata"]
                tag_names = meta.get("tag_names")
                continue
            records.append(obj)
            for comp in obj.get("completions", []):
                pos = comp.get("spacy_pos")
                if pos:
                    pos_list.append(pos)
    return tag_names, records, pos_list


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
    if total < 1:
        return None
    counts /= total
    return counts


def collect_samples(records: List[dict], tag_names: List[str], max_completions: int | None = None) -> Tuple[List[Sample], List[dict], int, int]:
    tag_to_idx = build_tag_map(tag_names)
    samples: List[Sample] = []
    infos: List[dict] = []
    skipped_unknown = 0
    skipped_single_pos = 0
    for rec in records:
        candidate = rec.get("candidate", {})
        completions = rec.get("completions", [])
        all_pos = [c.get("spacy_pos") for c in completions if c.get("spacy_pos") is not None]
        if any(pos not in tag_to_idx for pos in all_pos):
            skipped_unknown += 1
            continue
        if FILTER_SINGLE_POS:
            uniq = {pos for pos in all_pos if pos in tag_to_idx}
            if len(uniq) <= 1:
                skipped_single_pos += 1
                continue
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
    return samples, infos, skipped_unknown, skipped_single_pos


def encode_hidden_states(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for Gemma encoding.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    features = []
    for i in tqdm(range(0, len(samples), 64), desc="Encoding prefixes"):
        batch = samples[i : i + 64]
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


class LinearConeProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        cone_coords = torch.relu(self.linear(x))  # Project to positive cone
        return cone_coords / (cone_coords.sum(dim=-1, keepdim=True) + 1e-8)


def align_distributions(Y: np.ndarray, source_tags: List[str], target_tags: List[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(source_tags)}
    cols = [idx[t] for t in target_tags if t in idx]
    aligned = Y[:, cols]
    row_sums = np.clip(aligned.sum(axis=1, keepdims=True), 1e-8, None)
    return aligned / row_sums


def cache_or_encode(samples: List[Sample], cache_base: Path, expected_classes: int, tag_names: List[str]):
    x_path = cache_base.with_name(cache_base.name + "_X.npy")
    y_path = cache_base.with_name(cache_base.name + "_Y.npy")
    meta_path = cache_base.with_name(cache_base.name + "_meta.npz")

    if not FORCE_REGEN_CACHE and x_path.exists() and y_path.exists() and meta_path.exists():
        meta = np.load(meta_path, allow_pickle=True)
        cached_tags = meta.get("tag_names")
        if cached_tags is None:
            raise ValueError("Cache missing tag_names; regenerate.")
        if list(cached_tags) != tag_names:
            raise ValueError("Cache tag_names differ from expected; regenerate.")
        X_arr = np.load(x_path).astype(np.float32)
        Y_arr = np.load(y_path).astype(np.float32)
        if Y_arr.shape[1] != expected_classes:
            raise ValueError("Cache class count mismatch; regenerate.")
        print(f"Loaded cached activations from {cache_base}")
        return X_arr, Y_arr, tag_names

    X, Y = encode_hidden_states(samples)
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, X.astype(np.float16))
    np.save(y_path, Y.astype(np.float32))
    np.savez(meta_path, layers=np.array(LAYERS), tag_names=np.array(tag_names))
    print(f"Cached activations to {x_path} and {y_path}")
    return X.astype(np.float32), Y.astype(np.float32), tag_names


def sample_subset(X: np.ndarray, Y: np.ndarray, max_examples: int | None):
    if max_examples is None or len(X) <= max_examples:
        return X, Y
    idx = np.random.default_rng(SEED).choice(len(X), size=max_examples, replace=False)
    return X[idx], Y[idx]


def load_or_encode_all():
    # Train
    train_cache_x = CACHE_BASE.with_name(CACHE_BASE.name + "_X.npy")
    train_cache_y = CACHE_BASE.with_name(CACHE_BASE.name + "_Y.npy")
    train_cache_meta = CACHE_BASE.with_name(CACHE_BASE.name + "_meta.npz")

    train_tag_names = None
    train_X = train_Y = None
    train_samples: List[Sample] = []

    if not FORCE_REGEN_CACHE and train_cache_x.exists() and train_cache_y.exists() and train_cache_meta.exists():
        tmeta = np.load(train_cache_meta, allow_pickle=True)
        cached_tags = tmeta.get("tag_names")
        if cached_tags is None:
            raise ValueError("Train cache missing tag_names; regenerate.")
        train_tag_names = list(cached_tags)
        train_X = np.load(train_cache_x).astype(np.float32)
        train_Y = np.load(train_cache_y).astype(np.float32)
        print(f"Loaded cached activations from {CACHE_BASE}")

    if train_X is None or train_Y is None:
        tag_meta, train_records, train_pos = load_metadata_and_records(Path(INPUT_PATH))
        train_tag_names = tag_meta or sorted(set(train_pos))
        train_samples, _, skipped_unknown_train, skipped_single_train = collect_samples(
            train_records, train_tag_names, max_completions=NUM_TRAIN_RESAMPLES
        )
        if skipped_unknown_train:
            print(f"Skipped {skipped_unknown_train} train records due to unknown tags.")
        if skipped_single_train:
            print(f"Skipped {skipped_single_train} train records due to single POS.")
        if len(train_samples) == 0:
            raise ValueError("No train samples found after filtering.")
        train_X, train_Y, _ = cache_or_encode(train_samples, CACHE_BASE, expected_classes=len(train_tag_names), tag_names=train_tag_names)

    train_X, train_Y = sample_subset(train_X, train_Y, NUM_TRAIN_EXAMPLES)

    # Validation: external file; if TRAIN_RATIO < 1.0 we split internally
    val_cache_x = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_X.npy")
    val_cache_y = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_Y.npy")
    val_cache_meta = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_meta.npz")

    val_X = val_Y = None
    val_tag_names = None
    if TRAIN_RATIO >= 1.0 and not FORCE_REGEN_CACHE and val_cache_x.exists() and val_cache_y.exists() and val_cache_meta.exists():
        vmeta = np.load(val_cache_meta, allow_pickle=True)
        cached_vtags = vmeta.get("tag_names")
        if cached_vtags is None:
            raise ValueError("Val cache missing tag_names; regenerate.")
        val_tag_names = list(cached_vtags)
        val_X = np.load(val_cache_x).astype(np.float32)
        val_Y = np.load(val_cache_y).astype(np.float32)
        print(f"Loaded cached val activations from {VAL_CACHE_BASE}")

    if TRAIN_RATIO < 1.0:
        # internal split
        full_X, full_Y = train_X, train_Y
        train_len = int(len(full_X) * TRAIN_RATIO)
        val_len = len(full_X) - train_len
        train_subset, val_subset = random_split(
            TensorDataset(torch.from_numpy(full_X), torch.from_numpy(full_Y)),
            [train_len, val_len],
            generator=torch.Generator().manual_seed(SEED),
        )
        train_X = train_subset.dataset.tensors[0][train_subset.indices].numpy()
        train_Y = train_subset.dataset.tensors[1][train_subset.indices].numpy()
        val_X = val_subset.dataset.tensors[0][val_subset.indices].numpy()
        val_Y = val_subset.dataset.tensors[1][val_subset.indices].numpy()
        common_tags = train_tag_names
    else:
        if val_X is None or val_Y is None:
            val_meta, val_records, val_pos = load_metadata_and_records(Path(VAL_INPUT_PATH))
            val_tag_names = val_meta or sorted(set(val_pos))
            val_samples, _, skipped_unknown_val, skipped_single_val = collect_samples(val_records, val_tag_names, max_completions=None)
            if skipped_unknown_val:
                print(f"Skipped {skipped_unknown_val} val records due to unknown tags.")
            if skipped_single_val:
                print(f"Skipped {skipped_single_val} val records due to single POS.")
            if len(val_samples) == 0:
                raise ValueError("No val samples found after filtering.")
            val_X, val_Y, _ = cache_or_encode(val_samples, VAL_CACHE_BASE, expected_classes=len(val_tag_names), tag_names=val_tag_names)

        common_tags = sorted(set(train_tag_names) & set(val_tag_names))
        if not common_tags:
            raise ValueError("No overlapping tag names between train and val.")
        if set(train_tag_names) != set(common_tags):
            print(f"Dropping train-only tags: {sorted(set(train_tag_names) - set(common_tags))}")
            train_Y = align_distributions(train_Y, train_tag_names, common_tags)
        if set(val_tag_names) != set(common_tags):
            print(f"Dropping val-only tags: {sorted(set(val_tag_names) - set(common_tags))}")
            val_Y = align_distributions(val_Y, val_tag_names, common_tags)
        train_tag_names = common_tags

    return train_tag_names, train_X, train_Y, val_X, val_Y


def apply_class_subset(X: np.ndarray, Y: np.ndarray, tag_names: List[str], subset: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    missing = [t for t in subset if t not in tag_names]
    if missing:
        raise ValueError(f"Subset tags missing from tag_names: {missing}")
    idx = [tag_names.index(t) for t in subset]
    Y_sub = Y[:, idx]
    mask = (Y_sub.sum(axis=1) > 0) & ((Y_sub > 0).sum(axis=1) >= 2)
    dropped = int((~mask).sum())
    Y_sub = Y_sub[mask]
    X_sub = X[mask]
    row_sums = np.clip(Y_sub.sum(axis=1, keepdims=True), 1e-8, None)
    Y_sub = Y_sub / row_sums
    return X_sub, Y_sub, subset, dropped


def train_probe_with_val(X: np.ndarray, Y: np.ndarray, val_X: np.ndarray, val_Y: np.ndarray, *, batch_size: int, epochs: int, lr: float, warmup_steps: int, weight_decay: float):
    if val_X is None or val_Y is None:
        raise ValueError("Validation split is required for sweep.")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    val_dataset = TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_Y))

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cone_model = LinearConeProbe(X.shape[1], Y.shape[1]).cuda()
    cone_optimizer = torch.optim.AdamW(cone_model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    cosine_steps = max(1, total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cone_optimizer, T_max=cosine_steps, eta_min=0.0)

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

    best_cone_val = float("inf")
    best_cone_state = None
    last_cone_state = None

    for epoch in range(epochs):
        cone_model.train()
        train_losses = []
        train_r2_scores = []
        train_tv_scores = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train] bs={batch_size} lr={lr}")
        for step_idx, (xb, yb) in enumerate(pbar):
            xb = xb.cuda()
            yb = yb.cuda()
            cone_optimizer.zero_grad()
            cone_probs = cone_model(xb)
            loss = l2_loss(cone_probs, yb)
            loss.backward()
            cone_optimizer.step()
            global_step = epoch * len(train_loader) + step_idx
            if global_step < warmup_steps:
                warmup_lr = lr * float(global_step + 1) / float(max(1, warmup_steps))
                for g in cone_optimizer.param_groups:
                    g["lr"] = warmup_lr
            else:
                scheduler.step(global_step - warmup_steps)
            train_losses.append(loss.item())
            train_r2_scores.append(r2_score(cone_probs.detach(), yb).item())
            train_tv_scores.append(tv_distance(cone_probs.detach(), yb).item())

        cone_model.eval()
        with torch.no_grad():
            val_losses = []
            val_tvs = []
            val_r2s = []
            for xb, yb in val_loader:
                xb = xb.cuda()
                yb = yb.cuda()
                cone_probs = cone_model(xb)
                val_losses.append(l2_loss(cone_probs, yb).item())
                val_tvs.append(tv_distance(cone_probs, yb).item())
                val_r2s.append(r2_score(cone_probs, yb).item())
            avg_val = float(np.mean(val_losses))
            avg_tv = float(np.mean(val_tvs))
            avg_r2 = float(np.mean(val_r2s))
            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_r2 = float(np.mean(train_r2_scores)) if train_r2_scores else float("nan")
            train_tv = float(np.mean(train_tv_scores)) if train_tv_scores else float("nan")
            if avg_val < best_cone_val:
                best_cone_val = avg_val
                best_cone_state = copy.deepcopy(cone_model.state_dict())
            last_cone_state = copy.deepcopy(cone_model.state_dict())

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train L2: {train_mean:.4f} | train TV: {train_tv:.4f} | train R2: {train_r2:.4f} || "
            f"val L2: {avg_val:.4f} | val TV: {avg_tv:.4f} | val R2: {avg_r2:.4f}"
        )

    if SAVE_EPOCH == "best":
        cone_model.load_state_dict(best_cone_state)
    elif SAVE_EPOCH == "last":
        cone_model.load_state_dict(last_cone_state if last_cone_state else best_cone_state)
    else:
        raise ValueError(f"Unsupported SAVE_EPOCH: {SAVE_EPOCH}")

    def eval_split(loader):
        losses, tvs, r2s = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.cuda()
                yb = yb.cuda()
                cone_probs = cone_model(xb)
                losses.append(l2_loss(cone_probs, yb).item())
                tvs.append(tv_distance(cone_probs, yb).item())
                r2s.append(r2_score(cone_probs, yb).item())
        return float(np.mean(losses)), float(np.mean(tvs)), float(np.mean(r2s))

    train_loader_full = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_loss, train_tv, train_r2 = eval_split(train_loader_full)
    val_loss, val_tv, val_r2 = eval_split(val_loader)

    metrics = {
        "train_L2": train_loss,
        "train_TV": train_tv,
        "train_R2": train_r2,
        "val_L2": val_loss,
        "val_TV": val_tv,
        "val_R2": val_r2,
        "best_val_L2": best_cone_val,
    }
    return cone_model, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep distillation hyperparameters without reloading data.")
    parser.add_argument("--regen-cache", action="store_true", help="Force regeneration of cached activations")
    return parser.parse_args()


def save_probe(model: nn.Module, path: Path, hidden_dim: int, num_classes: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
        },
        path,
    )
    print(f"Saved probe to {path}")


def save_tag_names(tag_names: List[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(tag_names, f, ensure_ascii=False, indent=2)
    print(f"Saved tag names to {path}")


def main():
    args = parse_args()
    global FORCE_REGEN_CACHE
    if args.regen_cache:
        FORCE_REGEN_CACHE = True

    tag_names, train_X, train_Y, val_X, val_Y = load_or_encode_all()
    if val_X is None or val_Y is None:
        raise ValueError("Validation split is required for sweep.")

    if CLASS_SUBSET:
        print(f"Applying class subset: {CLASS_SUBSET}")
        train_X, train_Y, subset_tags, dropped_train = apply_class_subset(train_X, train_Y, tag_names, CLASS_SUBSET)
        val_X, val_Y, subset_tags_val, dropped_val = apply_class_subset(val_X, val_Y, tag_names, CLASS_SUBSET)
        tag_names = subset_tags
        print(f"Dropped {dropped_train} train rows and {dropped_val} val rows for subset filtering.")
        if len(train_X) == 0 or len(val_X) == 0:
            raise ValueError("No samples left after class subset filtering.")

    best_model = None
    best_metric = -float("inf")
    best_summary = None
    results = []

    for bs, lr, epochs in itertools.product(SWEEP_BATCH_SIZES, SWEEP_LRS, SWEEP_EPOCHS):
        print(f"\n=== Training combo: bs={bs}, lr={lr}, epochs={epochs} ===")
        model, metrics = train_probe_with_val(
            train_X,
            train_Y,
            val_X,
            val_Y,
            batch_size=bs,
            epochs=epochs,
            lr=lr,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
        )
        summary = {
            "batch_size": bs,
            "lr": lr,
            "epochs": epochs,
            **metrics,
        }
        results.append(summary)
        print(
            f"Result combo bs={bs} lr={lr} epochs={epochs}: "
            f"val L2={metrics['val_L2']:.4f}, val TV={metrics['val_TV']:.4f}, val R2={metrics['val_R2']:.4f}"
        )
        if metrics["val_R2"] > best_metric:
            best_metric = metrics["val_R2"]
            best_model = copy.deepcopy(model)
            best_summary = summary

    if best_model is None:
        raise RuntimeError("No sweep runs completed.")

    save_probe(best_model, LINEAR_CONE_OUTPUT_PATH, train_X.shape[1], len(tag_names))
    save_tag_names(tag_names, TAG_NAMES_PATH)
    print("\nBest combo:", best_summary)
    print("All results:")
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
