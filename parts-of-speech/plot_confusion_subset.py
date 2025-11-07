#!/usr/bin/env python
"""Re-plot a stored confusion matrix after dropping specific POS tags."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from pos_probe import constants, plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot confusion matrix excluding selected tags.")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics.json produced by train_probe.py",
    )
    parser.add_argument(
        "--drop-tags",
        type=str,
        default="INTJ",
        help="Comma-separated list of POS tags to remove from the plot (default: INTJ)",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "true", "pred", "all"],
        default="true",
        help="Normalization mode for the matrix (row=true, column=pred, all=global, none=counts).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the PNG. Defaults to metrics file name + '_subset.png'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_data = json.loads(args.metrics.read_text())
    matrix = np.asarray(metrics_data["confusion_matrix_softmax"])
    label_order = metrics_data.get("active_tags", constants.POS_TAGS)

    drop = {tag.strip().upper() for tag in args.drop_tags.split(",") if tag.strip()}
    unknown = drop.difference(constants.POS_TAGS)
    if unknown:
        raise ValueError(f"Unknown tags in --drop-tags: {', '.join(sorted(unknown))}")

    keep_indices = [idx for idx, tag in enumerate(label_order) if tag not in drop]
    if not keep_indices:
        raise ValueError("All tags were dropped; nothing to plot.")

    filtered_matrix = matrix[np.ix_(keep_indices, keep_indices)]
    filtered_labels = [label_order[idx] for idx in keep_indices]

    output_path = args.output or args.metrics.with_name(args.metrics.stem + "_subset.png")
    title = "Confusion Matrix"
    if drop:
        title += f" (without {', '.join(sorted(drop))})"

    plots.plot_confusion_matrix(
        filtered_matrix,
        filtered_labels,
        output_path,
        title=title,
        normalize=None if args.normalize == "none" else args.normalize,
    )
    print(f"Wrote filtered confusion matrix to {output_path}")


if __name__ == "__main__":
    main()
