import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from time import perf_counter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model / data configuration
MODEL_NAME = "google/gemma-2-2b"
DATASET_NAME = "karpathy/fineweb-edu-100b-shuffle"  # any HF dataset with a `text` column
TEXT_FIELD = "text"
SPLIT = "train"

# Entropy-based candidate selection
MAX_CONTEXT_TOKENS = 32
TOKENS_PER_EXAMPLE = 4  # how many targets to sample (weighted by entropy) per example
MAX_EXAMPLES = 400000  # optional cap over dataset items per shard

# Generation configuration
NUM_SAMPLES = 10
CHUNK_SIZE = 1516  # max total sequences per generate() call (after repetition)
MAX_NEW_TOKENS = 12
TEMPERATURE = 1.0
TOP_P = 0.995

# Output / reproducibility
OUTPUT_PREFIX = "data/entropy_resamples_shard"
RNG_SEED = 1234
DEBUG = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard-aware entropy-based resampler")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("GPU (CUDA or MPS) is required for resampling.")


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def compute_entropies(logits: torch.Tensor) -> np.ndarray:
    # logits: [seq_len, vocab]
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    ent = ent.float()  # ensure float32 before moving to CPU/NumPy
    return ent.cpu().numpy()


def weighted_sample_positions(entropies: np.ndarray, k: int) -> List[int]:
    if entropies.size == 0:
        return []
    weights = entropies.copy()
    weights[weights < 0] = 0
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()
    k = min(k, len(probs))
    return np.random.choice(len(probs), size=k, replace=False, p=probs).tolist()


def generate_for_candidates(
    candidates: List[dict],
    model,
    tokenizer,
    device: torch.device,
) -> List[List[dict]]:
    """
    Generate NUM_SAMPLES completions for each candidate prefix in a single (chunked) pass.
    Returns a list of completions lists, aligned with candidates.
    """
    if not candidates:
        return []

    pad_id = tokenizer.pad_token_id
    repeated: List[Tuple[int, List[int]]] = []
    for idx, cand in enumerate(candidates):
        prefix_ids = cand["prefix_token_ids"]
        for _ in range(NUM_SAMPLES):
            repeated.append((idx, prefix_ids))

    completions_acc: List[List[dict]] = [[] for _ in candidates]
    global_seq_idx = 0

    while global_seq_idx < len(repeated):
        chunk = repeated[global_seq_idx : global_seq_idx + CHUNK_SIZE]
        max_len = max(len(p[1]) for p in chunk)
        input_ids = []
        attn_masks = []
        for _, pref in chunk:
            pad_len = max_len - len(pref)
            input_ids.append([pad_id] * pad_len + pref)
            attn_masks.append([0] * pad_len + [1] * len(pref))

        input_tensor = torch.tensor(input_ids, device=device)
        attn_tensor = torch.tensor(attn_masks, device=device)

        chunk_seed = RNG_SEED + global_seq_idx
        torch.manual_seed(chunk_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(chunk_seed)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_tensor,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=pad_id,
            )

        outputs = outputs.cpu()
        prompt_len = max_len
        for out_row, (cand_idx, pref) in zip(outputs, chunk):
            new_tokens = out_row[prompt_len:]
            completions_acc[cand_idx].append(
                {
                    "token_ids": new_tokens.tolist(),
                    "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
                }
            )

        global_seq_idx += len(chunk)

    return completions_acc


def process_example(
    model,
    tokenizer,
    text: str,
    dataset_idx: int,
    device: torch.device,
) -> List[dict]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"][0]
    if input_ids.numel() <= 1:
        return []

    input_ids = input_ids[:MAX_CONTEXT_TOKENS]
    input_tensor = input_ids.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits.squeeze(0)  # [seq_len, vocab]

    # Exclude BOS prediction: logits for position i predict token i+1.
    target_logits = logits[:-1]
    entropies = compute_entropies(target_logits)
    # Positions correspond to target tokens 1..len-1
    target_positions = list(range(1, len(input_ids)))
    sampled_indices = weighted_sample_positions(entropies, TOKENS_PER_EXAMPLE)
    if DEBUG:
        sorted_ent = np.sort(entropies)
        def percentile(val):
            return float((sorted_ent <= val).sum() / len(sorted_ent) * 100.0)
        lines = []
        for i in sampled_indices:
            pos = target_positions[i]
            ent_val = float(entropies[i])
            pct = percentile(ent_val)
            lines.append(f"  pos={pos:>3d}, entropy={ent_val:6.3f}, percentile={pct:5.1f}")
        print("Sampled tokens (position / entropy / percentile):\n" + "\n".join(lines))

    candidates = []
    for idx in sampled_indices:
        target_pos = target_positions[idx]
        prefix_ids = input_ids[: target_pos + 1].tolist()
        target_id = int(input_ids[target_pos])
        target_text = tokenizer.decode([target_id], skip_special_tokens=True)
        candidates.append(
            {
                "dataset_idx": dataset_idx,
                "prefix_token_ids": prefix_ids,
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
                "target_position": target_pos,
                "target_token_id": target_id,
                "target_token_text": target_text,
                "entropy": float(entropies[idx]),
                "context_len": len(input_ids),
            }
        )
    return candidates


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard_index must be in [0, num_shards)")

    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    device = prepare_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    ensure_pad_token(tokenizer)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()

    # Streaming dataset; shard by modulo to avoid downloading the whole corpus.
    ds_iter = load_dataset(DATASET_NAME, split=SPLIT, streaming=True).shuffle(
        buffer_size=10_000, seed=RNG_SEED
    )
    shard = (row for idx, row in enumerate(ds_iter) if idx % args.num_shards == args.shard_index)
    if MAX_EXAMPLES is not None:
        shard = (row for _, row in zip(range(MAX_EXAMPLES), shard))
    total_examples = MAX_EXAMPLES if MAX_EXAMPLES is not None else None
    output_path = (
        args.output
        if args.output is not None
        else Path(f"{OUTPUT_PREFIX}{args.shard_index}.jsonl")
    )

    metadata = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "text_field": TEXT_FIELD,
        "split": SPLIT,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "tokens_per_example": TOKENS_PER_EXAMPLE,
        "num_samples_per_prefix": NUM_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_strategy": "modulo_streaming",
        "max_examples_per_shard": MAX_EXAMPLES,
        "rng_seed": RNG_SEED,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": metadata}, ensure_ascii=False) + "\n")
        load_time = 0.0
        forward_time = 0.0
        gen_time = 0.0
        t_loop_start = perf_counter()
        pending: List[Tuple[dict, str]] = []
        for dataset_idx, row in enumerate(
            tqdm(shard, desc=f"Shard {args.shard_index}: processing examples", total=total_examples)
        ):
            t_load = perf_counter()
            text = row.get(TEXT_FIELD, None)
            load_time += perf_counter() - t_load
            if not text:
                continue
            t_forward = perf_counter()
            candidates = process_example(model, tokenizer, text, dataset_idx, device)
            forward_time += perf_counter() - t_forward
            if not candidates:
                continue
            for cand in candidates:
                pending.append((cand, text[:400]))
            # Flush when repeated batch would exceed CHUNK_SIZE
            if len(pending) * NUM_SAMPLES >= CHUNK_SIZE:
                batch_cands = [c for c, _ in pending]
                t_gen = perf_counter()
                batch_comps = generate_for_candidates(batch_cands, model, tokenizer, device)
                gen_time += perf_counter() - t_gen
                for (cand, trunc_text), comps in zip(pending, batch_comps):
                    record = {
                        "candidate": cand,
                        "completions": comps,
                        "generation_params": {
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                            "max_new_tokens": MAX_NEW_TOKENS,
                            "num_samples": NUM_SAMPLES,
                        },
                        "truncated_text": trunc_text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                pending.clear()

        # Flush any remaining candidates
        if pending:
            batch_cands = [c for c, _ in pending]
            t_gen = perf_counter()
            batch_comps = generate_for_candidates(batch_cands, model, tokenizer, device)
            gen_time += perf_counter() - t_gen
            for (cand, trunc_text), comps in zip(pending, batch_comps):
                record = {
                    "candidate": cand,
                    "completions": comps,
                    "generation_params": {
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "num_samples": NUM_SAMPLES,
                    },
                    "truncated_text": trunc_text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_wall = perf_counter() - t_loop_start
        total_sections = load_time + forward_time + gen_time
        if total_sections == 0:
            total_sections = 1e-9
        print(
            "Timing (seconds / % of section total):\n"
            f"  load:     {load_time:.2f}  ({load_time/total_sections*100:.1f}%)\n"
            f"  forward:  {forward_time:.2f}  ({forward_time/total_sections*100:.1f}%)\n"
            f"  generate: {gen_time:.2f}  ({gen_time/total_sections*100:.1f}%)\n"
            f"  wall:     {total_wall:.2f}  (overall elapsed)"
        )


if __name__ == "__main__":
    main()
