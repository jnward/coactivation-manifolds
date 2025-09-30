"""Extract snippets where all selected features fire together."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect tokens where all features co-fire")
    parser.add_argument("run_dir", type=Path, help="Activation run directory with activations/ and metadata/")
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        required=True,
        help="Feature IDs to require (order determines activation vector layout)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of matched snippets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: metadata/probability_simplex/snippets_<features>.parquet)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    return parser.parse_args()


def resolve_output_path(run_dir: Path, features: List[int], user_path: Path | None) -> Path:
    if user_path is not None:
        return user_path
    tag = "_".join(str(fid) for fid in features)
    target_dir = run_dir / "metadata" / "probability_simplex"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"snippets_{tag}.parquet"


def collect_rows(
    activations_dir: Path,
    features: List[int],
    max_examples: int | None,
    *,
    show_progress: bool,
) -> dict[str, Iterable]:
    feature_set = set(features)
    doc_ids: List[int] = []
    token_indices: List[int] = []
    positions: List[int] = []
    token_texts: List[str] = []
    feature_lists: List[List[int]] = []
    activation_lists: List[List[float]] = []
    selected_vectors: List[List[float]] = []

    shard_paths = sorted((activations_dir).glob("shard=*"), key=lambda p: p.name)
    shard_iter = tqdm(shard_paths, desc="Shards", unit="shard") if show_progress else shard_paths

    total_seen = 0
    for shard_path in shard_iter:
        parquet_path = shard_path / "data.parquet"
        pf = pq.ParquetFile(parquet_path)
        inline_has_text = "token_text" in pf.schema.names
        sidecar_texts = None
        sidecar_index = 0
        if not inline_has_text:
            sidecar_path = shard_path / "token_text.parquet"
            if sidecar_path.exists():
                sidecar_table = pq.read_table(sidecar_path, columns=["token_text"])
                sidecar_texts = sidecar_table.column(0).to_pylist()
        batch_iter = pf.iter_batches(
            columns=["doc_id", "token_index", "position_in_doc", "token_text", "feature_ids", "activations"],
            use_threads=True,
        )
        for batch in batch_iter:
            doc_batch = batch.column("doc_id").to_numpy(zero_copy_only=False)
            token_batch = batch.column("token_index").to_numpy(zero_copy_only=False)
            pos_batch = batch.column("position_in_doc").to_numpy(zero_copy_only=False)
            if inline_has_text:
                text_batch = batch.column("token_text").to_pylist()
            elif sidecar_texts is not None:
                text_batch = sidecar_texts[sidecar_index : sidecar_index + batch.num_rows]
                sidecar_index += batch.num_rows
            else:
                text_batch = [""] * batch.num_rows
            feats_batch = batch.column("feature_ids").to_pylist()
            acts_batch = batch.column("activations").to_pylist()

            row_iter = zip(doc_batch, token_batch, pos_batch, text_batch, feats_batch, acts_batch)
            for doc_id, token_idx, pos, text, feats, acts in row_iter:
                if not feats:
                    continue
                feat_set = {int(fid) for fid in feats}
                if not feature_set.issubset(feat_set):
                    continue
                activation_map = {int(fid): float(act) for fid, act in zip(feats, acts)}
                vector = [activation_map.get(fid, 0.0) for fid in features]

                doc_ids.append(int(doc_id))
                token_indices.append(int(token_idx))
                positions.append(int(pos))
                token_texts.append(text)
                feature_lists.append([int(fid) for fid in feats])
                activation_lists.append([float(act) for act in acts])
                selected_vectors.append(vector)

                total_seen += 1
                if show_progress:
                    shard_iter.set_postfix(keep=total_seen)
                if max_examples is not None and total_seen >= max_examples:
                    break
            if max_examples is not None and total_seen >= max_examples:
                break
        if max_examples is not None and total_seen >= max_examples:
            break

    if not selected_vectors:
        raise RuntimeError("No tokens found where all requested features co-fire")

    if not any(str(text).strip() for text in token_texts):
        raise RuntimeError(
            "token_text column is empty; generate snippets first (e.g., run add_token_snippets.py)"
        )

    return {
        "doc_id": doc_ids,
        "token_index": token_indices,
        "position_in_doc": positions,
        "token_text": token_texts,
        "feature_ids": feature_lists,
        "activations": activation_lists,
        "selected_activation_vector": selected_vectors,
    }


def write_table(data: dict[str, Iterable], features: List[int], output_path: Path, run_dir: Path) -> None:
    table = pa.table(
        {
            "doc_id": pa.array(data["doc_id"], type=pa.int64()),
            "token_index": pa.array(data["token_index"], type=pa.int64()),
            "position_in_doc": pa.array(data["position_in_doc"], type=pa.int32()),
            "token_text": pa.array(data["token_text"], type=pa.large_string()),
            "feature_ids": pa.array(data["feature_ids"], type=pa.list_(pa.int32())),
            "activations": pa.array(data["activations"], type=pa.list_(pa.float32())),
            "selected_activation_vector": pa.array(data["selected_activation_vector"], type=pa.list_(pa.float32())),
        }
    )
    metadata = {
        b"features": json.dumps(features).encode(),
        b"run_dir": str(run_dir).encode(),
        b"example_count": str(table.num_rows).encode(),
    }
    table = table.replace_schema_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pa.parquet.write_table(table, output_path, compression="zstd")


def main() -> None:
    args = parse_args()
    features = list(dict.fromkeys(args.features))
    if not features:
        raise ValueError("At least one feature ID is required")

    run_dir = args.run_dir
    activations_dir = run_dir / "activations"

    data = collect_rows(
        activations_dir,
        features,
        args.max_examples,
        show_progress=not args.no_progress,
    )
    output_path = resolve_output_path(run_dir, features, args.output)
    write_table(data, features, output_path, run_dir)
    print(f"Wrote {len(data['selected_activation_vector'])} snippets to {output_path}")


if __name__ == "__main__":
    main()
