#!/usr/bin/env python
"""Generate token text snippets for existing activation shards."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add token_text sidecars to activation shards")
    parser.add_argument("run_dir", type=Path, help="Activation run directory containing activations/ and metadata/")
    parser.add_argument("dataset_path", type=Path, help="Path passed to datasets.load_from_disk for the source dataset")
    parser.add_argument("--split", default="train", help="Dataset split to read (default: train)")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--text-field", default="text", help="Field name in dataset containing raw text")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length used during logging")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size while rebuilding tokens")
    parser.add_argument("--window", type=int, default=10, help="Number of tokens of context on each side")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
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


def iterate_dataset(dataset, batch_size: int):
    batch: List[dict] = []
    for idx in range(len(dataset)):
        batch.append(dataset[idx])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    shards = load_shards(run_dir)
    row_counts = shard_row_counts(shards)
    total_tokens_required = sum(row_counts)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = load_from_disk(str(args.dataset_path))[args.split]

    progress = None
    if not args.no_progress:
        progress = tqdm(total=total_tokens_required, unit="tok", desc="Generating snippets")

    current_shard_idx = 0
    buffer: List[str] = []
    shard_target = row_counts[current_shard_idx]
    processed_tokens = 0

    def flush_buffer() -> None:
        nonlocal current_shard_idx, buffer, shard_target
        if len(buffer) == shard_target:
            write_sidecar(shards[current_shard_idx], buffer)
            current_shard_idx += 1
            buffer = []
            if current_shard_idx < len(row_counts):
                shard_target = row_counts[current_shard_idx]

    for batch in iterate_dataset(dataset, args.batch_size):
        tokenized = tokenizer(
            [sample[args.text_field] for sample in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = tokenized["input_ids"].cpu().numpy()
        attention_mask = tokenized.get("attention_mask")
        mask = attention_mask.cpu().numpy() if attention_mask is not None else None

        for b_idx, sample in enumerate(batch):
            seq_ids = input_ids[b_idx]
            seq_len = int(seq_ids.shape[0])
            seq_mask = mask[b_idx] if mask is not None else np.ones(seq_len, dtype=np.int64)

            for token_pos in range(seq_len):
                if processed_tokens >= total_tokens_required:
                    break
                if seq_mask[token_pos] == 0:
                    continue

                snippet = decode_snippet(tokenizer, seq_ids, token_pos, args.window)
                buffer.append(snippet)
                processed_tokens += 1
                if progress is not None:
                    progress.update(1)
                flush_buffer()

            if processed_tokens >= total_tokens_required or current_shard_idx >= len(row_counts):
                break

        if processed_tokens >= total_tokens_required or current_shard_idx >= len(row_counts):
            break

    if buffer:
        if current_shard_idx >= len(row_counts):
            raise RuntimeError("More snippets generated than activation rows")
        if len(buffer) != shard_target:
            raise RuntimeError("Final buffer does not match shard row count")
        write_sidecar(shards[current_shard_idx], buffer)

    if progress is not None:
        progress.close()

    if processed_tokens != total_tokens_required:
        raise RuntimeError(
            f"Generated {processed_tokens} snippets but activations expect {total_tokens_required}"
        )


if __name__ == "__main__":
    main()
