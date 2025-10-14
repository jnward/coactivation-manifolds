#!/usr/bin/env python
"""Compute pairwise feature coactivations from logged activation shards."""
from __future__ import annotations

import argparse
from pathlib import Path

from coactivation_manifolds.coactivation_stats import (
    compute_coactivation_counts,
    write_coactivation_table,
    write_feature_totals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Jaccard-ready coactivation counts")
    parser.add_argument("run_dir", type=Path, help="Directory containing activations/ and metadata/")
    parser.add_argument(
        "--first-token-idx",
        type=int,
        default=0,
        help="Inclusive starting token index per document (default: 0)",
    )
    parser.add_argument(
        "--last-token-idx",
        type=int,
        default=1024,
        help="Exclusive ending token index per document (default: 1024)",
    )
    parser.add_argument(
        "--output-name",
        default="coactivations.parquet",
        help="Filename for pairwise coactivation stats under metadata/",
    )
    parser.add_argument(
        "--feature-counts-name",
        default="feature_counts_trimmed.parquet",
        help="Filename for per-feature totals after token window filtering",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for shard processing (default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    counts = compute_coactivation_counts(
        run_dir,
        first_token_idx=args.first_token_idx,
        last_token_idx=args.last_token_idx,
        num_workers=args.num_workers,
    )

    metadata_dir = run_dir / "metadata"
    write_coactivation_table(counts, output_path=metadata_dir / args.output_name)
    write_feature_totals(
        counts.feature_counts,
        token_count=counts.token_count,
        first_token_idx=counts.first_token_idx,
        last_token_idx=counts.last_token_idx,
        output_path=metadata_dir / args.feature_counts_name,
    )


if __name__ == "__main__":
    main()
