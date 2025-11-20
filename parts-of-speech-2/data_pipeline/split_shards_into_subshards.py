import json
from pathlib import Path
from typing import List

import jsonlines
from tqdm import tqdm

# Configuration
INPUT_PREFIX = "data/entropy_resamples_shard"
INPUT_SUFFIX = ".jsonl"
NUM_INPUT_SHARDS = 7
NUM_SUBSHARDS = 28  # total output sub-shards
OUTPUT_PREFIX = "data/entropy_resamples_subshard"  # outputs will be OUTPUT_PREFIX{n}.jsonl


def collect_input_paths(prefix: str, suffix: str, num_shards: int) -> List[Path]:
    paths = []
    for i in range(num_shards):
        p = Path(f"{prefix}{i}{suffix}")
        if p.exists():
            paths.append(p)
        else:
            print(f"Warning: missing input shard {p}, skipping.")
    if not paths:
        raise ValueError("No input shards found.")
    return paths


def count_records(paths: List[Path]) -> int:
    total = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            next(f)  # skip metadata
            for _ in f:
                total += 1
    return total


def main():
    input_paths = collect_input_paths(INPUT_PREFIX, INPUT_SUFFIX, NUM_INPUT_SHARDS)

    writers = []
    out_paths = []
    for i in range(NUM_SUBSHARDS):
        out_path = Path(f"{OUTPUT_PREFIX}{i}.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_paths.append(out_path)
        writers.append(jsonlines.open(out_path, mode="w"))

    # Write metadata (from first available shard) to all outputs
    with input_paths[0].open("r", encoding="utf-8") as f:
        header = json.loads(next(f))
        for w in writers:
            w.write(header)

    total_records = count_records(input_paths)
    pbar = tqdm(total=total_records, desc="Splitting shards")

    target_idx = 0
    for path in input_paths:
        with path.open("r", encoding="utf-8") as f:
            next(f)  # skip metadata
            for line in f:
                obj = json.loads(line)
                writers[target_idx].write(obj)
                target_idx = (target_idx + 1) % NUM_SUBSHARDS
                pbar.update(1)

    for w in writers:
        w.close()
    pbar.close()

    print(f"Wrote {NUM_SUBSHARDS} sub-shards:")
    for p in out_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
