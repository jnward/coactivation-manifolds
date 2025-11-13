"""Coactivation counting utilities for SAE activation logs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass
class CoactivationCounts:
    """Sparse pairwise coactivation counts plus per-feature totals."""

    feature_counts: np.ndarray
    pair_counts: Dict[Tuple[int, int], int]
    token_count: int
    first_token_idx: int
    last_token_idx: int


def _process_shard_batch(
    shard_paths: List[Path],
    num_features: int,
    first_token_idx: int,
    last_token_idx: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], int]:
    """Process a batch of shards and return partial counts.

    Used by multiprocessing workers.
    """
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


def _process_shard_batch_with_len(
    args: Tuple[int, List[Path], int, int, int]
) -> Tuple[int, np.ndarray, Dict[Tuple[int, int], int], int]:
    """Wrapper to retain shard counts for progress reporting."""

    batch_len, shard_paths, num_features, first_token_idx, last_token_idx = args
    feature_counts, pair_counts, token_count = _process_shard_batch(
        shard_paths,
        num_features,
        first_token_idx,
        last_token_idx,
    )
    return batch_len, feature_counts, pair_counts, token_count


def _merge_results(
    results: List[Tuple[np.ndarray, Dict[Tuple[int, int], int], int]]
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], int]:
    """Merge results from multiple workers."""
    if not results:
        raise ValueError("No results to merge")

    # Merge feature counts (simple addition)
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


def compute_coactivation_counts(
    run_dir: Path | str,
    *,
    first_token_idx: int = 0,
    last_token_idx: int = 1024,
    num_workers: int = 1,
) -> CoactivationCounts:
    """Accumulate feature counts and pairwise intersections from activation shards.

    Args:
        run_dir: Directory containing activations/ and metadata/
        first_token_idx: Inclusive starting token index per document
        last_token_idx: Exclusive ending token index per document
        num_workers: Number of parallel workers (default: 1 for sequential processing)
    """
    run_path = Path(run_dir)
    activations_dir = run_path / "activations"
    metadata_dir = run_path / "metadata"

    feature_counts_path = metadata_dir / "feature_counts.parquet"
    if not feature_counts_path.exists():
        raise FileNotFoundError(f"Missing feature counts at {feature_counts_path}")

    counts_table = pq.read_table(feature_counts_path, columns=["count"])
    num_features = counts_table.num_rows

    if first_token_idx < 0:
        raise ValueError("first_token_idx must be non-negative")
    if last_token_idx <= first_token_idx:
        raise ValueError("last_token_idx must be greater than first_token_idx")

    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)

    if num_workers <= 1:
        # Sequential processing (original code path)
        feature_counts = np.zeros(num_features, dtype=np.int64)
        token_count = 0
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        for shard_path in tqdm(shard_paths, desc="Shards", leave=False):
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

        pair_counts_dict = dict(pair_counts)
    else:
        # Parallel processing
        # Distribute shards across workers in interleaved fashion
        worker_batches: List[List[Path]] = [[] for _ in range(num_workers)]
        for idx, shard_path in enumerate(shard_paths):
            worker_batches[idx % num_workers].append(shard_path)

        # Process batches in parallel
        print(f"Processing {len(shard_paths)} shards with {num_workers} workers...")
        tasks = [
            (len(batch), batch, num_features, first_token_idx, last_token_idx)
            for batch in worker_batches
            if batch
        ]

        print(f"Processing {len(shard_paths)} shards with {num_workers} workers...")
        with Pool(processes=num_workers) as pool, tqdm(
            total=len(shard_paths), desc="Shards", leave=False
        ) as progress:
            results = []
            for batch_len, feature_counts_part, pair_counts_part, token_count_part in pool.imap_unordered(
                _process_shard_batch_with_len, tasks
            ):
                progress.update(batch_len)
                results.append((feature_counts_part, pair_counts_part, token_count_part))

        # Merge results
        print("Merging results...")
        feature_counts, pair_counts_dict, token_count = _merge_results(results)

    return CoactivationCounts(
        feature_counts=feature_counts,
        pair_counts=pair_counts_dict,
        token_count=token_count,
        first_token_idx=first_token_idx,
        last_token_idx=last_token_idx,
    )


def write_coactivation_table(
    counts: CoactivationCounts,
    *,
    output_path: Path | str,
) -> None:
    """Persist coactivation intersections and derived Jaccard scores as Parquet."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    feature_counts = counts.feature_counts
    pair_items = list(counts.pair_counts.items())

    if pair_items:
        feature_i = []
        feature_j = []
        intersections = []
        count_i = []
        count_j = []
        jaccard = []

        for (a, b), inter in pair_items:
            total_i = feature_counts[a]
            total_j = feature_counts[b]
            if total_i == 0 or total_j == 0:
                continue
            union = total_i + total_j - inter
            if union == 0:
                continue
            feature_i.append(int(a))
            feature_j.append(int(b))
            intersections.append(int(inter))
            count_i.append(int(total_i))
            count_j.append(int(total_j))
            jaccard.append(float(inter / union))

        table = pa.table(
            {
                "feature_i": pa.array(feature_i, type=pa.int32()),
                "feature_j": pa.array(feature_j, type=pa.int32()),
                "intersection": pa.array(intersections, type=pa.int64()),
                "count_i": pa.array(count_i, type=pa.int64()),
                "count_j": pa.array(count_j, type=pa.int64()),
                "jaccard": pa.array(jaccard, type=pa.float32()),
            }
        )
    else:
        table = pa.table(
            {
                "feature_i": pa.array([], type=pa.int32()),
                "feature_j": pa.array([], type=pa.int32()),
                "intersection": pa.array([], type=pa.int64()),
                "count_i": pa.array([], type=pa.int64()),
                "count_j": pa.array([], type=pa.int64()),
                "jaccard": pa.array([], type=pa.float32()),
            }
        )

    pq.write_table(table, output, compression="zstd")


def write_feature_totals(
    feature_counts: np.ndarray,
    *,
    token_count: int,
    first_token_idx: int,
    last_token_idx: int,
    output_path: Path | str,
) -> None:
    """Persist per-feature activation totals to Parquet."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "feature_id": pa.array(np.arange(len(feature_counts)), type=pa.int32()),
            "count": pa.array(feature_counts, type=pa.int64()),
        }
    )
    metadata = {
        b"token_count": str(int(token_count)).encode(),
        b"first_token_idx": str(int(first_token_idx)).encode(),
        b"last_token_idx": str(int(last_token_idx)).encode(),
    }
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")
