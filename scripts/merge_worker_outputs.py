#!/usr/bin/env python
"""Merge outputs from multiple parallel workers into a unified activation directory."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge parallel worker outputs into unified structure"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory containing worker_*/ subdirectories",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Number of workers to merge",
    )
    parser.add_argument(
        "--keep-workers",
        action="store_true",
        help="Keep worker directories after merging (for debugging)",
    )
    return parser.parse_args()


def find_worker_dirs(output_dir: Path, num_workers: int) -> List[Path]:
    """Find and validate worker directories."""
    worker_dirs = []
    missing = []

    for i in range(num_workers):
        worker_dir = output_dir / f"worker_{i}"
        if not worker_dir.exists():
            missing.append(i)
        else:
            worker_dirs.append(worker_dir)

    if missing:
        print(f"ERROR: Missing worker directories: {missing}")
        sys.exit(1)

    return worker_dirs


def merge_activation_shards(output_dir: Path, worker_dirs: List[Path]) -> None:
    """Copy and renumber activation shards from all workers."""
    print("\nMerging activation shards...")

    merged_activations = output_dir / "activations"
    merged_activations.mkdir(parents=True, exist_ok=True)

    global_shard_idx = 0

    for worker_id, worker_dir in enumerate(worker_dirs):
        worker_activations = worker_dir / "activations"
        if not worker_activations.exists():
            print(f"WARNING: Worker {worker_id} has no activations directory")
            continue

        # Find all shards in this worker
        shard_dirs = sorted(worker_activations.glob("shard=*"), key=lambda p: int(p.name.split("=")[1]))

        print(f"  Worker {worker_id}: {len(shard_dirs)} shards")

        for shard_dir in tqdm(shard_dirs, desc=f"  Worker {worker_id}", leave=False):
            # Copy shard with new sequential numbering
            new_shard_dir = merged_activations / f"shard={global_shard_idx:06d}"
            shutil.copytree(shard_dir, new_shard_dir)
            global_shard_idx += 1

    print(f"  Total shards merged: {global_shard_idx}")


def merge_feature_counts(output_dir: Path, worker_dirs: List[Path]) -> None:
    """Merge feature counts from all workers."""
    print("\nMerging feature counts...")

    merged_metadata = output_dir / "metadata"
    merged_metadata.mkdir(parents=True, exist_ok=True)

    # Collect all feature count tables
    tables = []
    for worker_id, worker_dir in enumerate(worker_dirs):
        counts_path = worker_dir / "metadata" / "feature_counts.parquet"
        if not counts_path.exists():
            print(f"WARNING: Worker {worker_id} has no feature_counts.parquet")
            continue

        table = pq.read_table(counts_path)
        tables.append(table)
        print(f"  Worker {worker_id}: {table.num_rows} features")

    if not tables:
        print("ERROR: No feature count tables found")
        return

    # All workers should have same feature count
    feature_count = tables[0].num_rows
    for i, table in enumerate(tables):
        if table.num_rows != feature_count:
            print(f"ERROR: Worker {i} has {table.num_rows} features, expected {feature_count}")
            sys.exit(1)

    # Sum counts across workers
    count_arrays = [table.column("count").to_numpy() for table in tables]
    total_counts = np.sum(count_arrays, axis=0)

    # Write merged counts
    merged_table = pq.table({"count": total_counts})
    output_path = merged_metadata / "feature_counts.parquet"
    pq.write_table(merged_table, output_path)
    print(f"  Wrote: {output_path}")
    print(f"  Total activations across all features: {total_counts.sum()}")


def merge_feature_counts_trimmed(output_dir: Path, worker_dirs: List[Path]) -> None:
    """Merge trimmed feature counts if present."""
    print("\nMerging trimmed feature counts...")

    merged_metadata = output_dir / "metadata"

    # Check if any worker has trimmed counts
    has_trimmed = False
    for worker_dir in worker_dirs:
        if (worker_dir / "metadata" / "feature_counts_trimmed.parquet").exists():
            has_trimmed = True
            break

    if not has_trimmed:
        print("  No trimmed feature counts found (skipping)")
        return

    # Collect tables
    tables = []
    metadata_dicts = []

    for worker_id, worker_dir in enumerate(worker_dirs):
        counts_path = worker_dir / "metadata" / "feature_counts_trimmed.parquet"
        if not counts_path.exists():
            print(f"WARNING: Worker {worker_id} has no feature_counts_trimmed.parquet")
            continue

        table = pq.read_table(counts_path)
        tables.append(table)

        # Extract metadata if present
        if table.schema.metadata:
            metadata_dicts.append(table.schema.metadata)

    if not tables:
        print("  No trimmed counts to merge")
        return

    # Sum counts
    count_arrays = [table.column("count").to_numpy() for table in tables]
    total_counts = np.sum(count_arrays, axis=0)

    # Merge metadata (use first worker's metadata as template)
    merged_metadata_dict = metadata_dicts[0] if metadata_dicts else {}

    # Write merged table
    merged_table = pq.table({"count": total_counts})
    output_path = merged_metadata / "feature_counts_trimmed.parquet"
    pq.write_table(merged_table, output_path, metadata=merged_metadata_dict)
    print(f"  Wrote: {output_path}")


def rebuild_feature_index(output_dir: Path) -> None:
    """Rebuild feature index for merged activation shards."""
    print("\nRebuilding feature index...")

    # Import here to avoid circular dependency
    from coactivation_manifolds.activation_reader import FeatureIndexBuilder, FeatureIndexConfig

    # Read feature count to determine number of features
    feature_counts_path = output_dir / "metadata" / "feature_counts.parquet"
    if not feature_counts_path.exists():
        print("ERROR: No feature_counts.parquet found")
        return

    feature_counts_table = pq.read_table(feature_counts_path)
    num_features = feature_counts_table.num_rows

    # Build index
    activations_dir = output_dir / "activations"
    index_path = output_dir / "metadata" / "feature_index.parquet"

    config = FeatureIndexConfig(
        activations_dir=activations_dir,
        index_path=index_path,
        num_features=num_features,
    )

    builder = FeatureIndexBuilder(config)
    builder.build()

    print(f"  Wrote: {index_path}")


def cleanup_workers(output_dir: Path, num_workers: int) -> None:
    """Remove worker directories."""
    print("\nCleaning up worker directories...")

    for i in range(num_workers):
        worker_dir = output_dir / f"worker_{i}"
        if worker_dir.exists():
            shutil.rmtree(worker_dir)
            print(f"  Removed: worker_{i}/")


def main() -> None:
    args = parse_args()

    print("="*60)
    print("Merging Worker Outputs")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    print("="*60)

    # Find worker directories
    worker_dirs = find_worker_dirs(args.output_dir, args.num_workers)
    print(f"\nFound {len(worker_dirs)} worker directories")

    # Merge activation shards
    merge_activation_shards(args.output_dir, worker_dirs)

    # Merge metadata
    merge_feature_counts(args.output_dir, worker_dirs)
    merge_feature_counts_trimmed(args.output_dir, worker_dirs)

    # Rebuild feature index
    rebuild_feature_index(args.output_dir)

    # Cleanup
    if not args.keep_workers:
        cleanup_workers(args.output_dir, args.num_workers)
    else:
        print("\nKeeping worker directories (--keep-workers)")

    print("\n" + "="*60)
    print("SUCCESS: Merge complete!")
    print("="*60)
    print(f"Merged activations: {args.output_dir}/activations/")
    print(f"Merged metadata:    {args.output_dir}/metadata/")


if __name__ == "__main__":
    main()
