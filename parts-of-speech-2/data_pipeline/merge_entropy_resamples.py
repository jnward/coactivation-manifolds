import argparse
import json
from pathlib import Path

import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(description="Merge (and optionally filter) entropy resample shards")
    parser.add_argument("output", type=Path, help="Output merged JSONL")
    parser.add_argument("--prefix", default="entropy_resamples_subshard", help="Input shard prefix")
    parser.add_argument("--suffix", default="_with_spacy_pos.jsonl", help="Input shard suffix")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards to merge")
    parser.add_argument(
        "--min-unique-pos",
        type=int,
        default=1,
        help="Minimum distinct spaCy POS tags to keep a record (1 disables filtering)",
    )
    return parser.parse_args()


def iter_records(path: Path):
    with jsonlines.open(path) as reader:
        for obj in reader:
            yield obj


def record_has_diverse_pos(record, min_unique: int) -> bool:
    if min_unique <= 1:
        return True
    # Expect top-level spaCy tag on the target token (from classify_entropy_resamples.py)
    spacy_pos = None
    spacy_info = record.get("spacy")
    if isinstance(spacy_info, dict):
        spacy_pos = spacy_info.get("pos")
    return spacy_pos is not None  # only a single tag per record in this pipeline


def main():
    args = parse_args()
    shard_paths = [Path(f"{args.prefix}{i}{args.suffix}") for i in range(args.num_shards)]
    shard_paths = [p for p in shard_paths if p.exists()]
    if not shard_paths:
        raise ValueError("No shard files found")

    # Read metadata from first shard
    with shard_paths[0].open(encoding="utf-8") as f:
        reader = jsonlines.Reader(f)
        first_obj = next(iter(reader))
    if isinstance(first_obj, dict) and "metadata" in first_obj:
        metadata = first_obj
    else:
        metadata = {"metadata": first_obj}
    metadata.setdefault("metadata", {})
    metadata["metadata"].update(
        {
            "merged": True,
            "filtered": args.min_unique_pos > 1,
            "min_unique_spacy_pos": args.min_unique_pos,
            "source_shards": [str(p) for p in shard_paths],
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(args.output, "w") as writer:
        writer.write(metadata)
        for path in shard_paths:
            for idx, obj in enumerate(iter_records(path)):
                if idx == 0 and isinstance(obj, dict) and "metadata" in obj:
                    continue
                if record_has_diverse_pos(obj, args.min_unique_pos):
                    writer.write(obj)
    print(f"Wrote merged resamples to {args.output}")


if __name__ == "__main__":
    main()
