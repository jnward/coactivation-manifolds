#!/usr/bin/env python
"""Generate token text snippets for existing activation shards."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator, List
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from coactivation_manifolds.default_config import DEFAULT_MODEL_NAME

DEFAULT_DATASET = "monology/pile-uncopyrighted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add token_text sidecars to activation shards")
    parser.add_argument("run_dir", type=Path, help="Activation run directory containing activations/ and metadata/")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Path for datasets.load_from_disk (optional)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset name if --dataset-path is not provided")
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config name")
    parser.add_argument("--split", default="train", help="Dataset split to read (default: train)")
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL_NAME, help=f"Tokenizer name or path (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--text-field", default="text", help="Field containing raw text")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length used during logging")
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per batch while rebuilding tokens")
    parser.add_argument("--window", type=int, default=10, help="Number of tokens of context on each side")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--stream-dataset", action=argparse.BooleanOptionalAction, default=True, help="Stream dataset when using --dataset (default: True)")
    return parser.parse_args()


def load_shards(run_dir: Path) -> List[Path]:
    shard_dir = run_dir / "activations"
    if not shard_dir.exists():
        raise FileNotFoundError(f"No activations/ directory under {run_dir}")
    shards = sorted(shard_dir.glob("shard=*"), key=lambda p: p.name)
    if not shards:
        raise FileNotFoundError("No shard directories found")
    return shards


def shard_row_counts(shards: Iterable[Path]) -> List[int]:
    counts = []
    for shard in shards:
        pf = pq.ParquetFile(shard / "data.parquet")
        counts.append(pf.metadata.num_rows)
    return counts


def decode_snippet(tokenizer, seq_ids: np.ndarray, pos: int, window: int) -> str:
    ids = seq_ids.tolist()
    left_start = max(pos - window, 0)
    right_end = min(pos + window + 1, len(ids))

    left_ids = ids[left_start:pos]
    center_ids = ids[pos:pos + 1]
    right_ids = ids[pos + 1:right_end]

    decode = tokenizer.decode
    left_text = decode(left_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    center_text = decode(center_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    right_text = decode(right_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    snippet = f"{left_text} «{center_text}» {right_text}".strip()
    snippet = " ".join(snippet.split())
    return snippet


def write_sidecar(shard_dir: Path, snippets: List[str]) -> None:
    table = pa.table({"token_text": pa.array(snippets, type=pa.large_string())})
    pq.write_table(table, shard_dir / "token_text.parquet", compression="zstd", compression_level=5)


def iter_dataset(dataset, batch_size: int) -> Iterator[List[dict]]:
    batch: List[dict] = []
    if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        for idx in range(len(dataset)):
            batch.append(dataset[idx])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    else:
        for sample in dataset:
            batch.append(sample)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def maybe_flush(buffer: List[str], shards: List[Path], row_counts: List[int], shard_idx: int) -> int:
    target = row_counts[shard_idx]
    if len(buffer) == target:
        write_sidecar(shards[shard_idx], buffer)
        buffer.clear()
        return shard_idx + 1
    return shard_idx


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    shards = load_shards(run_dir)
    row_counts = shard_row_counts(shards)
    total_tokens_required = sum(row_counts)

    if args.dataset_path is not None:
        dataset = load_from_disk(str(args.dataset_path))[args.split]
    else:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
            streaming=args.stream_dataset,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    progress = None
    if not args.no_progress:
        progress = tqdm(total=total_tokens_required, unit="tok", desc="Snippets")

    current_shard_idx = 0
    buffer: List[str] = []
    processed_tokens = 0
    shard_target = row_counts[current_shard_idx]

    for batch in iter_dataset(dataset, args.batch_size):
        if processed_tokens >= total_tokens_required or current_shard_idx >= len(shards):
            break
        texts = [sample[args.text_field] for sample in batch]
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = tokenized["input_ids"].cpu().numpy()
        attention_mask = tokenized.get("attention_mask")
        mask = attention_mask.cpu().numpy() if attention_mask is not None else None

        for b_idx in range(len(batch)):
            seq_ids = input_ids[b_idx]
            seq_len = int(seq_ids.shape[0])
            seq_mask = mask[b_idx] if mask is not None else np.ones(seq_len, dtype=np.int64)
            for token_pos in range(seq_len):
                if processed_tokens >= total_tokens_required or current_shard_idx >= len(shards):
                    break
                if seq_mask[token_pos] == 0:
                    continue
                snippet = decode_snippet(tokenizer, seq_ids, token_pos, args.window)
                buffer.append(snippet)
                processed_tokens += 1
                if progress is not None:
                    progress.update(1)
                current_shard_idx = maybe_flush(buffer, shards, row_counts, current_shard_idx)
                if current_shard_idx < len(row_counts):
                    shard_target = row_counts[current_shard_idx]
            if processed_tokens >= total_tokens_required or current_shard_idx >= len(shards):
                break

    if buffer:
        if current_shard_idx >= len(shards):
            raise RuntimeError("Generated more snippets than activation rows")
        target = row_counts[current_shard_idx]
        if len(buffer) != target:
            raise RuntimeError("Final buffer does not match shard row count; dataset order mismatch")
        write_sidecar(shards[current_shard_idx], buffer)
        current_shard_idx += 1

    if progress is not None:
        progress.close()

    if processed_tokens != total_tokens_required:
        raise RuntimeError(
            f"Generated {processed_tokens} snippets but activations expect {total_tokens_required}" 
        )

    if current_shard_idx != len(shards):
        raise RuntimeError("Not all shards received snippets; dataset may be shorter than activations")


if __name__ == "__main__":
    main()
