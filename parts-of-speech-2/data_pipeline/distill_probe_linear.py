from __future__ import annotations

import argparse
import json
import html
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines as jsonlines
from sklearn.decomposition import IncrementalPCA

DATA_DIR = Path(__file__).parent / "data"
INPUT_PATH = DATA_DIR / "merged_shards.jsonl"
VAL_INPUT_PATH = DATA_DIR / "../../1000_merged.jsonl"
# INPUT_PATH = DATA_DIR / "../../filtered_train_resamples_with_spacy_pos.jsonl"
# VAL_INPUT_PATH = DATA_DIR / "../../filtered_val_resamples_with_spacy_pos.jsonl"
MODEL_NAME = "google/gemma-2-2b"
LAYERS = [0,2,4,6,8,10,12,14,16,18,20,22,24,26]
OUTPUT_PATH = Path("models/distilled_probe_gemma2b_decay.pt")
LINEAR_OUTPUT_PATH = Path("models/simple_linear_probe_gemma2b.pt")
TAG_NAMES_PATH = Path("models/distilled_probe_tag_names.json")
CACHE_BASE = Path("activation_cache/distill_cache_train_pca")
VAL_CACHE_BASE = Path("activation_cache/distill_cache_val_pca")
FORCE_REGEN_CACHE = False
TRAIN_RATIO = 1.0
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4
WARMUP_STEPS = 1000
REGULARIZATION_TYPE = "l2"  # "l2" or "l1"
REGULARIZATION_STRENGTH = 1e-4
MIN_COMPLETIONS = 1
SEED = 1234
SAVE_EPOCH = "last"  # "best" or "last"
NUM_TRAIN_RESAMPLES = None  # completions to use per train example (None for all)
NUM_TRAIN_EXAMPLES = None  # limit number of train examples (None for all)
FILTER_SINGLE_POS = True
# PCA settings (always applied)
PCA_DIM = 1024  # per-layer components
PCA_BATCH_SIZE = 64  # forward batch size for PCA fit/transform
TRUNCATE_PCA_DIM = 1024  # if set and < PCA_DIM, slice per-layer dims from cached X


def effective_pca_dim() -> int:
    if TRUNCATE_PCA_DIM is not None and TRUNCATE_PCA_DIM < PCA_DIM:
        return TRUNCATE_PCA_DIM
    return PCA_DIM


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
    if total < MIN_COMPLETIONS:
        return None
    counts /= total
    return counts


def collect_samples(records: List[dict], tag_names: List[str], max_completions: int | None = None) -> Tuple[List[Sample], List[dict], int]:
    tag_to_idx = build_tag_map(tag_names)
    samples: List[Sample] = []
    infos: List[dict] = []
    skipped_unknown = 0
    skipped_single_pos = 0
    for rec in records:
        candidate = rec.get("candidate", {})
        completions = rec.get("completions", [])
        # Skip records containing tags outside tag_names
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


def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()
    return model, tokenizer


def iterate_layer_batches(samples: List[Sample], model, tokenizer, batch_size: int):
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
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
        layer_batches = []
        for layer_idx in LAYERS:
            layer_tensor = hidden_states[layer_idx].float()  # (batch, seq, hidden)
            vecs = []
            for b_idx, s in enumerate(batch):
                pos = len(s.tokens) - 1
                vecs.append(layer_tensor[b_idx, pos].cpu().numpy())
            layer_batches.append(np.stack(vecs, axis=0))
        yield layer_batches


def fit_pcas(samples: List[Sample], model, tokenizer):
    eff_dim = effective_pca_dim()
    pcas = [IncrementalPCA(n_components=eff_dim) for _ in LAYERS]
    total_batches = math.ceil(len(samples) / PCA_BATCH_SIZE) if len(samples) else 0
    # Accumulate until we have at least PCA_DIM samples per layer before first partial_fit
    accum = [ [] for _ in LAYERS ]
    for layer_batches in tqdm(iterate_layer_batches(samples, model, tokenizer, PCA_BATCH_SIZE), total=total_batches, desc="Fitting PCA"):
        for li, batch_vecs in enumerate(layer_batches):
            accum[li].append(batch_vecs)
        ready = all(sum(chunk.shape[0] for chunk in acc_list) >= PCA_DIM for acc_list in accum)
        if ready:
            for li, acc_list in enumerate(accum):
                stacked = np.concatenate(acc_list, axis=0)
                pcas[li].partial_fit(stacked)
                accum[li] = []
    # Fit on any remaining small accumulators
    for li, acc_list in enumerate(accum):
        if acc_list:
            stacked = np.concatenate(acc_list, axis=0)
            pcas[li].partial_fit(stacked)
    return pcas


def transform_with_pca(samples: List[Sample], pcas, model, tokenizer):
    feats = []
    total_batches = math.ceil(len(samples) / PCA_BATCH_SIZE) if len(samples) else 0
    for layer_batches in tqdm(iterate_layer_batches(samples, model, tokenizer, PCA_BATCH_SIZE), total=total_batches, desc="Encoding with PCA"):
        reduced_chunks = []
        for li, batch_vecs in enumerate(layer_batches):
            comp = pcas[li]["components"]
            mean = pcas[li]["mean"]
            reduced = (batch_vecs - mean) @ comp.T
            reduced_chunks.append(reduced)
        concat = np.concatenate(reduced_chunks, axis=1)
        feats.append(concat.astype(np.float32))
    X = np.vstack(feats) if feats else np.zeros((0, len(LAYERS) * PCA_DIM), dtype=np.float32)
    eff_dim = effective_pca_dim()
    if X.shape[1] != len(LAYERS) * eff_dim:
        # Slice if PCA components > effective dim
        per_layer = X.shape[1] // len(LAYERS)
        X = X.reshape(X.shape[0], len(LAYERS), per_layer)[:, :, :eff_dim].reshape(X.shape[0], len(LAYERS) * eff_dim)
    Y = np.stack([s.distribution for s in samples]).astype(np.float32)
    return X, Y


def save_pcas(pcas, cache_base: Path):
    for li, p in enumerate(pcas):
        path = cache_base.with_name(f"{cache_base.name}_pca_layer{li}.npz")
        np.savez(path, components=p.components_, mean=p.mean_)


def load_pcas(cache_base: Path, num_layers: int):
    pcas = []
    for li in range(num_layers):
        path = cache_base.with_name(f"{cache_base.name}_pca_layer{li}.npz")
        if not path.exists():
            return None
        arr = np.load(path)
        pcas.append({"components": arr["components"], "mean": arr["mean"]})
    return pcas


def encode_hidden_states(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    model, tokenizer = get_model_and_tokenizer()
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


class SimpleLinearProbe(nn.Module):
    """Simple linear probe without ReLU or normalization."""
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # Direct linear output


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
        target_safe = target_probs
        return 0.5 * torch.sum(torch.abs(probs - target_safe), dim=-1).mean()

    # Train simple linear probe (no ReLU or normalization)
    linear_model = SimpleLinearProbe(X.shape[1], Y.shape[1]).cuda()
    if REGULARIZATION_TYPE == "l1":
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=LR)
    else:
        optimizer = torch.optim.AdamW(linear_model.parameters(), lr=LR, weight_decay=REGULARIZATION_STRENGTH)
    total_steps = EPOCHS * len(train_loader)
    cosine_steps = max(1, total_steps - WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=0.0)
    best_val_loss = float("inf")
    best_state = None
    last_state = None
    for epoch in range(EPOCHS):
        linear_model.train()
        train_losses = []
        train_r2_scores = []
        train_tv_scores = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
        for step_idx, (xb, yb) in enumerate(pbar):
            xb = xb.cuda()
            yb = yb.cuda()
            optimizer.zero_grad()
            preds = linear_model(xb)
            base_loss = l2_loss(preds, yb)
            loss = base_loss

            # Compute weight sparsity
            with torch.no_grad():
                total_weights = sum(p.numel() for n, p in linear_model.named_parameters() if 'weight' in n)
                zero_weights = sum((p == 0).sum().item() for n, p in linear_model.named_parameters() if 'weight' in n)
                sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

            if REGULARIZATION_TYPE == "l1":
                l1_reg = sum(p.abs().sum() for name, p in linear_model.named_parameters() if 'weight' in name)
                reg_loss = REGULARIZATION_STRENGTH * l1_reg
                loss = loss + reg_loss
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} L2={base_loss.item():.4f} Reg={reg_loss.item():.4f} Sp={sparsity:.1%}")
            else:
                pbar.set_description(f"Epoch {epoch+1}/{EPOCHS} L2={base_loss.item():.4f} Sp={sparsity:.1%}")
            loss.backward()
            optimizer.step()
            # Warmup then cosine
            global_step = epoch * len(train_loader) + step_idx
            if scheduler:
                if global_step < WARMUP_STEPS:
                    warmup_lr = LR * float(global_step + 1) / float(max(1, WARMUP_STEPS))
                    for g in optimizer.param_groups:
                        g["lr"] = warmup_lr
                else:
                    scheduler.step(global_step - WARMUP_STEPS)
            train_losses.append(loss.item())
            train_r2_scores.append(r2_score(preds.detach(), yb).item())
            train_tv_scores.append(tv_distance(preds.detach(), yb).item())

        linear_model.eval()
        with torch.no_grad():
            val_losses = []
            val_tvs = []
            val_r2s = []
            for xb, yb in val_loader:
                xb = xb.cuda()
                yb = yb.cuda()
                preds = linear_model(xb)
                val_losses.append(l2_loss(preds, yb).item())
                val_tvs.append(tv_distance(preds, yb).item())
                val_r2s.append(r2_score(preds, yb).item())
            avg_val = float(np.mean(val_losses))
            avg_tv = float(np.mean(val_tvs))
            avg_r2 = float(np.mean(val_r2s))
            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            train_r2 = float(np.mean(train_r2_scores)) if train_r2_scores else float("nan")
            train_tv = float(np.mean(train_tv_scores)) if train_tv_scores else float("nan")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = copy.deepcopy(linear_model.state_dict())
            last_state = copy.deepcopy(linear_model.state_dict())

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train L2: {train_mean:.4f} | train TV: {train_tv:.4f} | train R2: {train_r2:.4f} || "
            f"val L2: {avg_val:.4f} | val TV: {avg_tv:.4f} | val R2: {avg_r2:.4f}"
        )

    if SAVE_EPOCH == "best":
        linear_model.load_state_dict(best_state)
    elif SAVE_EPOCH == "last":
        linear_model.load_state_dict(last_state if last_state else best_state)
    else:
        raise ValueError(f"Unsupported SAVE_EPOCH: {SAVE_EPOCH}")

    # Compute metrics on train
    linear_model.eval()
    def eval_split(loader):
        losses, tvs, r2s = [], [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.cuda()
                yb = yb.cuda()
                preds = linear_model(xb)
                losses.append(l2_loss(preds, yb).item())
                tvs.append(tv_distance(preds, yb).item())
                r2s.append(r2_score(preds, yb).item())
        return float(np.mean(losses)), float(np.mean(tvs)), float(np.mean(r2s))

    train_loader_full = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=BATCH_SIZE)
    train_loss, train_tv, train_r2 = eval_split(train_loader_full)

    val_loss, val_tv, val_r2 = eval_split(val_loader)

    print(f"Simple Linear Probe - Train L2: {train_loss:.4f}, TV: {train_tv:.4f}, R2: {train_r2:.4f}")
    print(f"Simple Linear Probe - Best val L2: {best_val_loss:.4f}")
    print(f"Simple Linear Probe - Val L2: {val_loss:.4f}, TV: {val_tv:.4f}, R2: {val_r2:.4f}")

    return softmax_model, best_softmax_state, linear_model, val_loss, val_tv, val_r2, train_loss, train_tv, train_r2


def cache_or_encode(samples: List[Sample], cache_base: Path, expected_classes: int, tag_names: List[str] | None, pca_models=None):
    x_path = cache_base.with_name(cache_base.name + "_X.npy")
    y_path = cache_base.with_name(cache_base.name + "_Y.npy")
    meta_path = cache_base.with_name(cache_base.name + "_meta.npz")

    cached_tag_names = None
    if x_path.exists() and y_path.exists() and meta_path.exists() and not FORCE_REGEN_CACHE:
        meta = np.load(meta_path, allow_pickle=True)
        layers_ok = list(meta.get("layers", [])) == LAYERS
        cached_tag_names = meta.get("tag_names")
        cached_pca = bool(meta.get("apply_pca", False))
        cached_pca_dim = int(meta.get("pca_dim", 0)) if "pca_dim" in meta else None
        Y_arr = np.load(y_path)
        if layers_ok and Y_arr.shape[1] == expected_classes and cached_tag_names is not None and list(cached_tag_names) == (tag_names or [] ) and cached_pca and cached_pca_dim == PCA_DIM:
            X_arr = np.load(x_path)
            X_arr = truncate_features(X_arr)
            print(f"Loaded cached activations from {cache_base} (X/Y files)")
            return X_arr.astype(np.float32), Y_arr.astype(np.float32), cached_tag_names
        else:
            print("Cache configuration mismatch; recomputing activations.")

    if pca_models is None:
        model, tokenizer = get_model_and_tokenizer()
        pca_models = fit_pcas(samples, model, tokenizer)
        save_pcas(pca_models, cache_base)
        pca_models = [{"components": p.components_, "mean": p.mean_} for p in pca_models]
    else:
        model, tokenizer = get_model_and_tokenizer()
    X, Y = transform_with_pca(samples, pca_models, model, tokenizer)
    X = truncate_features(X)
    cache_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_path, X.astype(np.float32))
    np.save(y_path, Y.astype(np.float32))
    np.savez(meta_path, layers=np.array(LAYERS), tag_names=np.array(tag_names) if tag_names else None, apply_pca=True, pca_dim=PCA_DIM)
    print(f"Cached activations to {x_path} and {y_path}")
    return X.astype(np.float32), Y.astype(np.float32), tag_names


def align_distributions(Y: np.ndarray, source_tags: List[str], target_tags: List[str]) -> np.ndarray:
    """Select and renormalize columns to match target_tags ordering."""
    src_idx = {t: i for i, t in enumerate(source_tags)}
    cols = [src_idx[t] for t in target_tags if t in src_idx]
    aligned = Y[:, cols]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-8, None)
    return aligned / row_sums


def sample_subset(X: np.ndarray, Y: np.ndarray, max_examples: int | None):
    if max_examples is None or len(X) <= max_examples:
        return X, Y
    idx = np.random.default_rng(SEED).choice(len(X), size=max_examples, replace=False)
    return X[idx], Y[idx]


def truncate_features(X: np.ndarray) -> np.ndarray:
    eff_dim = effective_pca_dim()
    if eff_dim is None:
        return X
    # Determine current per-layer width from shape
    if X.shape[1] % len(LAYERS) != 0:
        raise ValueError("X shape not divisible by number of layers; cannot truncate.")
    per_layer = X.shape[1] // len(LAYERS)
    if per_layer <= eff_dim:
        return X
    num_layers = len(LAYERS)
    return X.reshape(X.shape[0], num_layers, per_layer)[:, :, :eff_dim].reshape(X.shape[0], num_layers * eff_dim)


def load_or_encode_all():
    # Train data
    train_tag_names = None
    train_X = train_Y = None
    train_samples: List[Sample] = []
    train_infos: List[dict] = []
    train_indices = None
    val_indices = None

    train_cache_x = CACHE_BASE.with_name(CACHE_BASE.name + "_X.npy")
    train_cache_y = CACHE_BASE.with_name(CACHE_BASE.name + "_Y.npy")
    train_cache_meta = CACHE_BASE.with_name(CACHE_BASE.name + "_meta.npz")

    if not FORCE_REGEN_CACHE and train_cache_x.exists() and train_cache_y.exists() and train_cache_meta.exists():
        meta = np.load(train_cache_meta, allow_pickle=True)
        cached_tags = meta.get("tag_names")
        if cached_tags is None:
            raise ValueError("Train cache is missing tag_names; regenerate cache.")
        cached_pca = bool(meta.get("apply_pca", False))
        cached_pca_dim = int(meta.get("pca_dim", 0)) if "pca_dim" in meta else None
        layers_ok = list(meta.get("layers", [])) == LAYERS
        if cached_pca and cached_pca_dim == PCA_DIM and layers_ok:
            train_tag_names = list(cached_tags)
            train_X = np.load(train_cache_x).astype(np.float32)
            train_Y = np.load(train_cache_y).astype(np.float32)
            print(f"Loaded cached activations from {CACHE_BASE}")

    if train_X is None or train_Y is None:
        tag_meta, train_records, train_pos = load_metadata_and_records(Path(INPUT_PATH))
        if train_tag_names is None:
            train_tag_names = tag_meta or sorted(set(train_pos))
            if tag_meta is None:
                print(f"No tag_names in train metadata; inferred from train records: {train_tag_names}")
        train_samples, train_infos, skipped_unknown_train, skipped_single_train = collect_samples(
            train_records, train_tag_names, max_completions=NUM_TRAIN_RESAMPLES
        )
        if skipped_unknown_train:
            print(f"Skipped {skipped_unknown_train} train records due to unknown tags.")
        if skipped_single_train:
            print(f"Skipped {skipped_single_train} train records due to single POS.")
        print(f"Collected {len(train_samples)} train samples.")
        if len(train_samples) == 0:
            raise ValueError("No train samples found after filtering.")
        train_X, train_Y, _ = cache_or_encode(train_samples, CACHE_BASE, expected_classes=len(train_tag_names), tag_names=train_tag_names, pca_models=None)
        train_X = truncate_features(train_X)

    train_X, train_Y = sample_subset(train_X, train_Y, NUM_TRAIN_EXAMPLES)

    # Validation handling
    val_X = val_Y = None
    val_samples: List[Sample] = []
    val_infos: List[dict] = []
    val_internal_X = val_internal_Y = None

    if TRAIN_RATIO < 1.0:
        # Internal split
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
        val_internal_X = val_subset.dataset.tensors[0][val_subset.indices].numpy()
        val_internal_Y = val_subset.dataset.tensors[1][val_subset.indices].numpy()
        val_X, val_Y = val_internal_X, val_internal_Y
        train_indices = list(train_subset.indices)
        val_indices = list(val_subset.indices)
    else:
        val_cache_x = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_X.npy")
        val_cache_y = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_Y.npy")
        val_cache_meta = VAL_CACHE_BASE.with_name(VAL_CACHE_BASE.name + "_meta.npz")

        val_tag_names = None
        if not FORCE_REGEN_CACHE and val_cache_x.exists() and val_cache_y.exists() and val_cache_meta.exists():
            vmeta = np.load(val_cache_meta, allow_pickle=True)
            cached_vtags = vmeta.get("tag_names")
            if cached_vtags is None:
                raise ValueError("Val cache is missing tag_names; regenerate cache.")
            cached_pca = bool(vmeta.get("apply_pca", False))
            cached_pca_dim = int(vmeta.get("pca_dim", 0)) if "pca_dim" in vmeta else None
            layers_ok = list(vmeta.get("layers", [])) == LAYERS
            if cached_pca and cached_pca_dim == PCA_DIM and layers_ok:
                val_tag_names = list(cached_vtags)
                val_X = np.load(val_cache_x).astype(np.float32)
                val_Y = np.load(val_cache_y).astype(np.float32)
                val_X = truncate_features(val_X)
                print(f"Loaded cached val activations from {VAL_CACHE_BASE}")

        if val_X is None or val_Y is None:
            val_meta, val_records, val_pos = load_metadata_and_records(Path(VAL_INPUT_PATH))
            val_tag_names = val_meta or sorted(set(val_pos))
            val_samples, val_infos, skipped_unknown_val, skipped_single_val = collect_samples(
                val_records, val_tag_names, max_completions=None
            )
            if skipped_unknown_val:
                print(f"Skipped {skipped_unknown_val} val records due to unknown tags.")
            if skipped_single_val:
                print(f"Skipped {skipped_single_val} val records due to single POS.")
            if len(val_samples) == 0:
                raise ValueError("No val samples found after filtering.")
            pcas = load_pcas(CACHE_BASE, len(LAYERS))
            if pcas is None:
                raise ValueError("Train PCA not found; regenerate train cache first.")
            val_X, val_Y, _ = cache_or_encode(val_samples, VAL_CACHE_BASE, expected_classes=len(val_tag_names), tag_names=val_tag_names, pca_models=pcas)

        # Align tag sets
        if val_tag_names is None:
            val_tag_names = train_tag_names
        common_tags = sorted(set(train_tag_names) & set(val_tag_names))
        if not common_tags:
            raise ValueError("No overlapping tag names between train and val.")
        train_only = sorted(set(train_tag_names) - set(common_tags))
        val_only = sorted(set(val_tag_names) - set(common_tags))
        if train_only:
            print(f"Dropping train-only tags: {train_only}")
            train_Y = align_distributions(train_Y, train_tag_names, common_tags)
        if val_only:
            print(f"Dropping val-only tags: {val_only}")
            val_Y = align_distributions(val_Y, val_tag_names, common_tags)
        train_tag_names = common_tags

    # Final truncation in case cache path was hit without slicing
    train_X = truncate_features(train_X)
    if val_X is not None:
        val_X = truncate_features(val_X)

    return (
        train_tag_names,
        train_X,
        train_Y,
        val_internal_X,
        val_internal_Y,
        val_X,
        val_Y,
        train_samples,
        val_samples,
        val_infos,
        train_infos,
        train_indices,
        val_indices,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Distill probe from resampled activations")
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
    (
        tag_names,
        train_X,
        train_Y,
        val_internal_X,
        val_internal_Y,
        val_X,
        val_Y,
        train_samples,
        val_samples,
        val_infos,
        train_infos,
        train_indices,
        val_indices,
    ) = load_or_encode_all()

    softmax_model, best_softmax_state, linear_model, val_loss, val_tv, val_r2, train_loss, train_tv, train_r2 = train_probe_with_val(
        train_X, train_Y, val_X, val_Y
    )

    # Save models
    save_probe(linear_model, LINEAR_OUTPUT_PATH, train_X.shape[1], len(tag_names))
    save_probe(softmax_model, OUTPUT_PATH, train_X.shape[1], len(tag_names)) if softmax_model else None
    save_tag_names(tag_names, TAG_NAMES_PATH)

    print(f"Simple Linear Probe - Train L2: {train_loss:.4f}, TV: {train_tv:.4f}, R2: {train_r2:.4f}")
    print(f"Simple Linear Probe - Val L2: {val_loss:.4f}, TV: {val_tv:.4f}, R2: {val_r2:.4f}")


if __name__ == "__main__":
    main()
