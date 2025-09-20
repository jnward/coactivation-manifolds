"""Coactivation counting utilities for SAE activation logs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

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


def compute_coactivation_counts(
    run_dir: Path | str,
    *,
    ignore_prefix_tokens: int = 100,
) -> CoactivationCounts:
    """Accumulate feature counts and pairwise intersections from activation shards."""

    run_path = Path(run_dir)
    activations_dir = run_path / "activations"
    metadata_dir = run_path / "metadata"

    feature_counts_path = metadata_dir / "feature_counts.parquet"
    if not feature_counts_path.exists():
        raise FileNotFoundError(f"Missing feature counts at {feature_counts_path}")

    counts_table = pq.read_table(feature_counts_path, columns=["count"])
    num_features = counts_table.num_rows
    feature_counts = np.zeros(num_features, dtype=np.int64)
    token_count = 0

    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)
    for shard_path in tqdm(shard_paths, desc="Shards", leave=False):
        file_path = shard_path / "data.parquet"
        table = pq.read_table(file_path, columns=["position_in_doc", "feature_ids"])
        positions: Iterable[int] = table.column("position_in_doc").to_pylist()
        feature_lists: Iterable[Iterable[int]] = table.column("feature_ids").to_pylist()

        for position, features in zip(positions, feature_lists):
            if position < ignore_prefix_tokens:
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

    return CoactivationCounts(
        feature_counts=feature_counts,
        pair_counts=dict(pair_counts),
        token_count=token_count,
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
    metadata = {b"token_count": str(int(token_count)).encode()}
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")
