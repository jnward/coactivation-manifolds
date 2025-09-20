#!/usr/bin/env python
"""Plot Jaccard similarity percentiles on a log-log scale."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Jaccard similarity percentiles")
    parser.add_argument(
        "coactivations_path",
        type=Path,
        help="Path to coactivations.parquet with a jaccard column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (defaults to coactivations_percentiles.png next to the data)",
    )
    parser.add_argument(
        "--min_tail",
        type=float,
        default=1e-4,
        help="Smallest top-percentile slice to plot (default: 1e-4 => 0.0001%%)",
    )
    parser.add_argument(
        "--max_tail",
        type=float,
        default=1.0,
        help="Largest top-percentile slice to plot (default: 1 => top 1%%)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=400,
        help="Number of percentile samples (default: 400)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.coactivations_path, columns=["jaccard"])
    jaccard = table.column(0).to_numpy(zero_copy_only=False)

    tail_percents = np.logspace(
        np.log10(args.max_tail),
        np.log10(args.min_tail),
        num=args.points,
    )
    percentiles = 100 - tail_percents
    values = np.percentile(jaccard, percentiles)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tail_percents, values, marker="o", markersize=2, linewidth=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Top percentile (100 - percentile)")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("Jaccard tail percentiles")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    xticks = [1.0, 0.1, 0.01, 0.001, 0.0001]
    ticks_in_range = [tick for tick in xticks if args.min_tail <= tick <= args.max_tail]
    if ticks_in_range:
        ax.set_xticks(ticks_in_range)
        ax.set_xticklabels([f"{tick:g}%" for tick in ticks_in_range])

    output = Path.cwd() / "coactivations_percentiles.png" if args.output is None else args.output
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Wrote plot to {output}")


if __name__ == "__main__":
    main()
