#!/usr/bin/env python
"""Profile activation pipeline to identify performance bottlenecks."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

from coactivation_manifolds.default_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER,
    DEFAULT_SAE_RELEASE,
    DEFAULT_SAE_NAME,
)
from coactivation_manifolds.sae_loader import load_sae


class Timer:
    """Simple context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile activation pipeline")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--dataset", default="monology/pile-uncopyrighted", help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    print("="*80)
    print("ACTIVATION PIPELINE PROFILER")
    print("="*80)
    print(f"Model: {DEFAULT_MODEL_NAME}")
    print(f"SAE: {DEFAULT_SAE_RELEASE} / {DEFAULT_SAE_NAME}")
    print(f"Batches: {args.num_batches}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # Track timings
    timings = {
        "dataset_fetch": [],
        "tokenization_encode": [],
        "model_forward": [],
        "sae_encode": [],
        "token_decode": [],
        "record_building": [],
    }

    # Detailed record building timings
    detailed_timings = {
        "feature_extract": [],
        "cache_lookup": [],
        "string_join": [],
        "strip": [],
        "split_join": [],
    }

    # Load dataset
    print("Loading dataset...")
    with Timer("dataset_load") as t:
        dataset = load_dataset(args.dataset, split=args.split, streaming=True)
        dataset_iter = iter(dataset)
    print(f"  Dataset loaded in {t.elapsed:.2f}s\n")

    # Load tokenizer
    print("Loading tokenizer...")
    with Timer("tokenizer_load") as t:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    print(f"  Tokenizer loaded in {t.elapsed:.2f}s\n")

    # Load model
    print("Loading model...")
    with Timer("model_load") as t:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )
        model.eval()
    print(f"  Model loaded in {t.elapsed:.2f}s\n")

    # Load SAE
    print("Loading SAE...")
    with Timer("sae_load") as t:
        sae_handle = load_sae(
            sae_release=DEFAULT_SAE_RELEASE,
            sae_name=DEFAULT_SAE_NAME,
            device=args.device,
        )
        sae_handle.sae.eval()
    print(f"  SAE loaded in {t.elapsed:.2f}s\n")

    print("="*80)
    print("PROFILING BATCHES")
    print("="*80 + "\n")

    device = torch.device(args.device)

    for batch_idx in range(args.num_batches):
        print(f"\n--- Batch {batch_idx + 1}/{args.num_batches} ---")

        # 1. Fetch batch from dataset
        with Timer("fetch") as t:
            batch_samples = []
            for _ in range(args.batch_size):
                try:
                    sample = next(dataset_iter)
                    batch_samples.append(sample)
                except StopIteration:
                    break
            if not batch_samples:
                print("  Dataset exhausted")
                break
            texts = [s["text"] for s in batch_samples]
        timings["dataset_fetch"].append(t.elapsed)
        print(f"  1. Dataset fetch: {t.elapsed:.4f}s")

        # 2. Tokenization (encoding)
        with Timer("tokenize") as t:
            tokenized = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
        timings["tokenization_encode"].append(t.elapsed)
        print(f"  2. Tokenization:  {t.elapsed:.4f}s")

        # 3. Model forward pass
        with Timer("forward") as t:
            with torch.no_grad():
                outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states[DEFAULT_LAYER]
        timings["model_forward"].append(t.elapsed)
        print(f"  3. Model forward: {t.elapsed:.4f}s")

        # 4. SAE encoding
        with Timer("sae") as t:
            with torch.no_grad():
                sae_outputs = sae_handle.sae.encode(hidden_states)
                sae_outputs = torch.relu(sae_outputs)
        timings["sae_encode"].append(t.elapsed)
        print(f"  4. SAE encode:    {t.elapsed:.4f}s")

        # 5. Token decode (batch_decode)
        with Timer("decode") as t:
            input_ids = tokenized["input_ids"].detach().cpu()
            unique_ids = torch.unique(input_ids)
            unique_ids_list = unique_ids.tolist()
            decoded_tokens = tokenizer.batch_decode(
                [[tid] for tid in unique_ids_list],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_cache = dict(zip(unique_ids_list, decoded_tokens))
        timings["token_decode"].append(t.elapsed)
        print(f"  5. Token decode:  {t.elapsed:.4f}s ({len(token_cache)} unique tokens)")

        # 6. Record building (simulate snippet construction)
        with Timer("records") as t:
            sae_cpu = sae_outputs.detach().cpu()
            attention_mask = tokenized.get("attention_mask")
            mask_cpu = attention_mask.detach().cpu() if attention_mask is not None else None

            num_records = 0

            # Sub-timings for this batch
            batch_detailed = {k: 0.0 for k in detailed_timings.keys()}

            for batch_idx_inner in range(len(batch_samples)):
                seq_ids = input_ids[batch_idx_inner]
                seq_acts = sae_cpu[batch_idx_inner]
                seq_len = int(seq_ids.shape[0])
                mask = mask_cpu[batch_idx_inner] if mask_cpu is not None else torch.ones(seq_len, dtype=torch.long)

                for token_pos in range(seq_len):
                    if mask[token_pos] == 0:
                        continue

                    # Time: feature extraction
                    t0 = time.time()
                    feature_vector = seq_acts[token_pos]
                    nz = torch.nonzero(feature_vector, as_tuple=False).squeeze(-1)
                    batch_detailed["feature_extract"] += time.time() - t0

                    # Snippet construction
                    window = 10
                    left_start = max(token_pos - window, 0)
                    right_end = min(token_pos + window + 1, seq_len)
                    seq_ids_np = seq_ids.numpy()

                    # Time: cache lookups
                    t0 = time.time()
                    left_tokens = [token_cache[int(tid)] for tid in seq_ids_np[left_start:token_pos]]
                    center_token = token_cache[int(seq_ids_np[token_pos])]
                    right_tokens = [token_cache[int(tid)] for tid in seq_ids_np[token_pos + 1 : right_end]]
                    batch_detailed["cache_lookup"] += time.time() - t0

                    # Time: string joining
                    t0 = time.time()
                    left_text = ''.join(left_tokens)
                    right_text = ''.join(right_tokens)
                    snippet = f"{left_text} «{center_token}» {right_text}"
                    batch_detailed["string_join"] += time.time() - t0

                    # Time: strip
                    t0 = time.time()
                    snippet = snippet.strip()
                    batch_detailed["strip"] += time.time() - t0

                    # Time: split/join
                    t0 = time.time()
                    snippet = " ".join(snippet.split())
                    batch_detailed["split_join"] += time.time() - t0

                    num_records += 1

            # Accumulate detailed timings
            for key in detailed_timings.keys():
                detailed_timings[key].append(batch_detailed[key])

        timings["record_building"].append(t.elapsed)
        print(f"  6. Record build:  {t.elapsed:.4f}s ({num_records} records)")
        print(f"     - Feature extract: {batch_detailed['feature_extract']:.4f}s")
        print(f"     - Cache lookup:    {batch_detailed['cache_lookup']:.4f}s")
        print(f"     - String join:     {batch_detailed['string_join']:.4f}s")
        print(f"     - Strip:           {batch_detailed['strip']:.4f}s")
        print(f"     - Split/join:      {batch_detailed['split_join']:.4f}s")

    # Print summary
    print("\n" + "="*80)
    print("TIMING SUMMARY (averaged over batches)")
    print("="*80)

    total_avg = 0
    for name, times in timings.items():
        if times:
            avg = np.mean(times)
            std = np.std(times)
            total_avg += avg
            print(f"{name:25s}: {avg:.4f}s ± {std:.4f}s")

    print(f"{'TOTAL per batch':25s}: {total_avg:.4f}s")
    print(f"\nEstimated time for 1M tokens (~500 batches): {total_avg * 500 / 3600:.2f} hours")

    # Show percentages
    print("\n" + "="*80)
    print("TIME BREAKDOWN")
    print("="*80)
    for name, times in timings.items():
        if times:
            avg = np.mean(times)
            pct = (avg / total_avg) * 100
            print(f"{name:25s}: {pct:5.1f}%")
    print("="*80)

    # Show detailed record building breakdown
    print("\n" + "="*80)
    print("DETAILED RECORD BUILDING BREAKDOWN")
    print("="*80)
    record_avg = np.mean(timings["record_building"])
    for name, times in detailed_timings.items():
        if times:
            avg = np.mean(times)
            std = np.std(times)
            pct = (avg / record_avg) * 100
            print(f"{name:25s}: {avg:.4f}s ± {std:.4f}s ({pct:5.1f}% of record building)")
    print("="*80)


if __name__ == "__main__":
    main()
