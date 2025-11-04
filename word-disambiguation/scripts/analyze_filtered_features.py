#!/usr/bin/env python3
"""Summarize passed features from a filter_features CSV."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV produced by filter_features.py",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=1,
        help="Only show word groups with at least this many features (default: 1)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only display the first N groups after sorting",
    )
    return parser.parse_args()


def load_passed_features(csv_path: Path) -> List[Tuple[str, int]]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    groups: List[Tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "feature_id",
            "canonical_token",
            "passed",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            if row.get("passed") not in {"1", "true", "True"}:
                continue
            token = (row.get("canonical_token") or "").strip()
            if not token:
                continue
            feature_id = int(row["feature_id"])
            groups.append((token, feature_id))
    return groups


def group_by_token(pairs: List[Tuple[str, int]]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for token, fid in pairs:
        grouped[token].append(fid)
    return grouped


def main() -> None:
    args = parse_args()
    pairs = load_passed_features(args.csv_path)
    grouped = group_by_token(pairs)

    sorted_groups = sorted(
        grouped.items(), key=lambda item: len(item[1]), reverse=True
    )

    print(f"Total tokens with passing features: {len(sorted_groups)}")
    print(f"Total passing features: {len(pairs)}")
    print()
    header = f"{'Token':20} | {'#Features':9} | Feature IDs"
    print(header)
    print("-" * len(header))

    shown = 0
    for token, feature_ids in sorted_groups:
        count = len(feature_ids)
        if count < args.min_size:
            continue
        print(f"{token:20} | {count:9d} | {', '.join(map(str, feature_ids))}")
        shown += 1
        if args.top is not None and shown >= args.top:
            break


if __name__ == "__main__":
    main()
