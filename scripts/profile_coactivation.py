#!/usr/bin/env python
"""Profile coactivation computation to identify performance bottlenecks."""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool
from typing import List


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


def _process_shard_batch(
    shard_paths: List[Path],
    num_features: int,
    first_token_idx: int,
    last_token_idx: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], int]:
    """Process a batch of shards and return partial counts."""
    feature_counts = np.zeros(num_features, dtype=np.int64)
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    token_count = 0

    for shard_path in shard_paths:
        file_path = shard_path / "data.parquet"
        table = pq.read_table(file_path, columns=["position_in_doc", "feature_ids"])
        positions: Iterable[int] = table.column("position_in_doc").to_pylist()
        feature_lists: Iterable[Iterable[int]] = table.column("feature_ids").to_pylist()

        for position, features in zip(positions, feature_lists):
            if position < first_token_idx or position >= last_token_idx:
                continue
            token_count += 1
            feats = [int(f) for f in features]
            if not feats:
                continue
            for fid in feats:
                feature_counts[fid] += 1
            if len(feats) < 2:
                continue
            for a, b in combinations(feats, 2):
                if a > b:
                    a, b = b, a
                pair_counts[(a, b)] += 1

    return feature_counts, dict(pair_counts), token_count


def _merge_results(
    results: List[Tuple[np.ndarray, Dict[Tuple[int, int], int], int]]
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], int]:
    """Merge results from multiple workers."""
    if not results:
        raise ValueError("No results to merge")

    # Merge feature counts
    merged_feature_counts = results[0][0].copy()
    for feature_counts, _, _ in results[1:]:
        merged_feature_counts += feature_counts

    # Merge pair counts dictionaries
    merged_pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for _, pair_counts, _ in results:
        for pair, count in pair_counts.items():
            merged_pair_counts[pair] += count

    # Sum token counts
    total_token_count = sum(token_count for _, _, token_count in results)

    return merged_feature_counts, dict(merged_pair_counts), total_token_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile coactivation computation")
    parser.add_argument("run_dir", type=Path, help="Directory containing activations/ and metadata/")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards to profile")
    parser.add_argument("--first-token-idx", type=int, default=0, help="First token position (inclusive)")
    parser.add_argument("--last-token-idx", type=int, default=1024, help="Last token position (exclusive)")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*80)
    print("COACTIVATION COMPUTATION PROFILER")
    print("="*80)
    print(f"Run directory: {args.run_dir}")
    print(f"Token window: [{args.first_token_idx}, {args.last_token_idx})")
    print(f"Shards to profile: {args.num_shards}")
    print(f"Number of workers: {args.num_workers}")
    print("="*80 + "\n")

    # Get number of features
    metadata_dir = args.run_dir / "metadata"
    feature_counts_path = metadata_dir / "feature_counts.parquet"
    if not feature_counts_path.exists():
        print(f"ERROR: {feature_counts_path} not found")
        sys.exit(1)

    counts_table = pq.read_table(feature_counts_path, columns=["count"])
    num_features = counts_table.num_rows
    print(f"Number of features: {num_features}\n")

    # Get shard paths
    activations_dir = args.run_dir / "activations"
    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)
    shards_to_process = shard_paths[:args.num_shards]

    if len(shards_to_process) < args.num_shards:
        print(f"WARNING: Only {len(shards_to_process)} shards available")

    print(f"Processing {len(shards_to_process)} shards...\n")

    # Overall timing
    overall_start = time.time()

    if args.num_workers > 1:
        # PARALLEL PROCESSING
        print(f"Using {args.num_workers} parallel workers\n")

        # Distribute shards across workers
        worker_batches: List[List[Path]] = [[] for _ in range(args.num_workers)]
        for idx, shard_path in enumerate(shards_to_process):
            worker_batches[idx % args.num_workers].append(shard_path)

        # Process in parallel
        with Timer("parallel_processing") as t:
            with Pool(processes=args.num_workers) as pool:
                tasks = [
                    (batch, num_features, args.first_token_idx, args.last_token_idx)
                    for batch in worker_batches if batch
                ]
                results = pool.starmap(_process_shard_batch, tasks)

        print(f"Parallel processing completed in {t.elapsed:.2f}s")

        # Merge results
        with Timer("merge") as t:
            feature_counts, pair_counts_dict, total_tokens_processed = _merge_results(results)
        print(f"Merging completed in {t.elapsed:.2f}s\n")

        pair_counts = pair_counts_dict
        total_tokens = total_tokens_processed  # Simplified for parallel case

        # No detailed per-shard timing in parallel mode
        timings = {}

    else:
        # SEQUENTIAL PROCESSING (with detailed timing)
        print("Using sequential processing (detailed timing)\n")

        # Timing accumulators
        timings = {
            "read_parquet": [],
            "to_pylist": [],
            "position_filter": [],
            "type_conversion": [],
            "feature_counting": [],
            "pairwise_combinations": [],
        }

        feature_counts = np.zeros(num_features, dtype=np.int64)
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        total_tokens = 0
        total_tokens_processed = 0

        for shard_idx, shard_path in enumerate(shards_to_process):
            print(f"--- Shard {shard_idx + 1}/{len(shards_to_process)} ---")
            file_path = shard_path / "data.parquet"

            # 1. Read parquet
            with Timer("read") as t:
                table = pq.read_table(file_path, columns=["position_in_doc", "feature_ids"])
            timings["read_parquet"].append(t.elapsed)
            print(f"  1. Read parquet:    {t.elapsed:.4f}s ({table.num_rows} rows)")

            # 2. Convert to Python lists
            with Timer("pylist") as t:
                positions: Iterable[int] = table.column("position_in_doc").to_pylist()
                feature_lists: Iterable[Iterable[int]] = table.column("feature_ids").to_pylist()
            timings["to_pylist"].append(t.elapsed)
            print(f"  2. to_pylist():     {t.elapsed:.4f}s")

            # 3. Position filtering
            shard_time_filter = 0.0
            shard_time_convert = 0.0
            shard_time_count = 0.0
            shard_time_pairs = 0.0
            shard_tokens = 0
            shard_tokens_processed = 0

            for position, features in zip(positions, feature_lists):
                shard_tokens += 1

                # Time: position filtering
                t0 = time.time()
                if position < args.first_token_idx or position >= args.last_token_idx:
                    shard_time_filter += time.time() - t0
                    continue
                shard_time_filter += time.time() - t0

                shard_tokens_processed += 1

                # Time: type conversion
                t0 = time.time()
                feats = [int(f) for f in features]
                shard_time_convert += time.time() - t0

                if not feats:
                    continue

                # Time: feature counting
                t0 = time.time()
                for fid in feats:
                    feature_counts[fid] += 1
                shard_time_count += time.time() - t0

                if len(feats) < 2:
                    continue

                # Time: pairwise combinations
                t0 = time.time()
                for a, b in combinations(feats, 2):
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] += 1
                shard_time_pairs += time.time() - t0

            timings["position_filter"].append(shard_time_filter)
            timings["type_conversion"].append(shard_time_convert)
            timings["feature_counting"].append(shard_time_count)
            timings["pairwise_combinations"].append(shard_time_pairs)

            total_tokens += shard_tokens
            total_tokens_processed += shard_tokens_processed

            print(f"  3. Position filter: {shard_time_filter:.4f}s ({shard_tokens_processed}/{shard_tokens} tokens kept)")
            print(f"  4. Type conversion: {shard_time_convert:.4f}s")
            print(f"  5. Feature count:   {shard_time_count:.4f}s")
            print(f"  6. Pairwise combos: {shard_time_pairs:.4f}s")

    # Calculate overall time
    overall_elapsed = time.time() - overall_start

    # Print summary
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Total wall-clock time: {overall_elapsed:.2f}s")
    print(f"Total tokens processed: {total_tokens_processed}")
    print(f"Total feature pairs found: {len(pair_counts)}")
    print("="*80)

    if timings:
        # Detailed timing summary (only for sequential mode)
        print("\n" + "="*80)
        print("TIMING SUMMARY (averaged over shards)")
        print("="*80)

        total_avg = 0
        for name, times in timings.items():
            if times:
                avg = np.mean(times)
                std = np.std(times)
                total_avg += avg
                print(f"{name:25s}: {avg:.4f}s ± {std:.4f}s")

        print(f"{'TOTAL per shard':25s}: {total_avg:.4f}s")

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


if __name__ == "__main__":
    main()
