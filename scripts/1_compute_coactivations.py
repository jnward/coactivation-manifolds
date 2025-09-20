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
        "--ignore-prefix",
        type=int,
        default=100,
        help="Tokens to skip at the start of each document (default: 100)",
    )
    parser.add_argument(
        "--output-name",
        default="coactivations.parquet",
        help="Filename for pairwise coactivation stats under metadata/",
    )
    parser.add_argument(
        "--feature-counts-name",
        default="feature_counts_trimmed.parquet",
        help="Filename for per-feature totals after prefix skipping",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    counts = compute_coactivation_counts(run_dir, ignore_prefix_tokens=args.ignore_prefix)

    metadata_dir = run_dir / "metadata"
    write_coactivation_table(counts, output_path=metadata_dir / args.output_name)
    write_feature_totals(
        counts.feature_counts,
        token_count=counts.token_count,
        output_path=metadata_dir / args.feature_counts_name,
    )


if __name__ == "__main__":
    main()
