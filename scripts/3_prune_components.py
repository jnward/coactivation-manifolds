#!/usr/bin/env python
"""Compute connected components after pruning coactivation edges by Jaccard/cosine."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq

from coactivation_manifolds.sae_loader import (
    DEFAULT_SAE_NAME,
    DEFAULT_SAE_RELEASE,
    load_sae,
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        rank = self.rank
        parent = self.parent
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune coactivation graph and count components")
    parser.add_argument(
        "coactivations_path",
        type=Path,
        help="Path to coactivations.parquet",
    )
    parser.add_argument(
        "feature_counts_path",
        type=Path,
        help="Path to feature_counts_trimmed.parquet",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Minimum Jaccard similarity to keep an edge (default: 0.8)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="Maximum cosine similarity to keep an edge (optional)",
    )
    parser.add_argument(
        "--decoder-path",
        type=Path,
        default=None,
        help="Decoder matrix (.npy/.npz). When omitted, the SAE is loaded and cached to metadata/decoder_directions.npy",
    )
    parser.add_argument(
        "--sae-release",
        default=DEFAULT_SAE_RELEASE,
        help="sae-lens release identifier (used when generating decoder directions)",
    )
    parser.add_argument(
        "--sae-name",
        default=DEFAULT_SAE_NAME,
        help="sae-lens SAE identifier within the release",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the SAE on when generating decoder directions (default: cpu)",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=None,
        help="Remove features whose activation rate exceeds this fraction of tokens",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Row batch size when streaming the Parquet file (default: 1e6)",
    )
    return parser.parse_args()


def _to_numpy(matrix) -> np.ndarray:
    candidate = matrix
    if hasattr(candidate, "weight"):
        candidate = candidate.weight
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    array = np.asarray(candidate)
    if array.ndim != 2:
        raise ValueError("Decoder weights must be 2D")
    return array.astype(np.float32, copy=False)


def _load_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        if not data.files:
            raise ValueError(f"Empty npz archive at {path}")
        array = data[data.files[0]]
        data.close()
    else:
        array = data
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Decoder weights must be 2D")
    return arr


def _extract_decoder_matrix(sae) -> np.ndarray:
    for attr in ("decoder", "W_dec", "decoder_weight"):
        candidate = getattr(sae, attr, None)
        if candidate is None:
            continue
        try:
            return _to_numpy(candidate)
        except ValueError:
            continue
    raise ValueError("Unable to locate decoder weights on SAE")


def _resolve_decoder_vectors(
    *,
    feature_count: int,
    decoder_path: Path | None,
    metadata_dir: Path,
    sae_release: str,
    sae_name: str,
    device: str,
) -> Tuple[np.ndarray, Path]:
    target_path = Path(decoder_path) if decoder_path is not None else metadata_dir / "decoder_directions.npy"
    if target_path.exists():
        vectors = _load_array(target_path)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        handle = load_sae(sae_release=sae_release, sae_name=sae_name, device=device)
        decoder = _extract_decoder_matrix(handle.sae)
        if decoder.shape[0] == feature_count:
            vectors = decoder
        elif decoder.shape[1] == feature_count:
            vectors = decoder.T
        else:
            raise ValueError("Decoder matrix does not match feature count")
        vectors = vectors.astype(np.float32, copy=False)
        np.save(target_path, vectors)
        print(f"Saved decoder directions to {target_path}")
    if vectors.shape[0] != feature_count:
        if vectors.shape[1] == feature_count:
            vectors = vectors.T
        else:
            raise ValueError("Decoder matrix does not cover all feature IDs")
    return vectors.astype(np.float32, copy=False), target_path


def main() -> None:
    args = parse_args()

    counts_table = pq.read_table(args.feature_counts_path, columns=["feature_id", "count"])
    metadata = counts_table.schema.metadata or {}
    token_count_raw = metadata.get(b"token_count")
    if token_count_raw is None:
        raise ValueError(
            "feature_counts_trimmed.parquet missing token_count metadata; rerun 1_compute_coactivations.py"
        )
    token_count = int(token_count_raw.decode())
    feature_ids = counts_table.column("feature_id").to_numpy(zero_copy_only=False)
    counts = counts_table.column("count").to_numpy(zero_copy_only=False)

    if len(feature_ids) == 0:
        print("No features found in counts table")
        return

    max_feature_id = int(feature_ids.max())
    uf = UnionFind(max_feature_id + 1)

    active_mask = np.zeros(max_feature_id + 1, dtype=bool)
    density_threshold = args.density_threshold
    removed_for_density = 0
    for fid, count in zip(feature_ids, counts):
        idx = int(fid)
        if count > 0:
            if idx >= active_mask.size:
                continue
            if density_threshold is not None and token_count > 0:
                density = float(count) / float(token_count)
                if density > density_threshold:
                    removed_for_density += 1
                    continue
            active_mask[idx] = True
    active_indices = np.nonzero(active_mask)[0]
    active_features = set(int(i) for i in active_indices)

    cosine_threshold = args.cosine_threshold
    vectors = None
    norms = None
    if cosine_threshold is not None:
        metadata_dir = args.feature_counts_path.resolve().parent
        vectors, _ = _resolve_decoder_vectors(
            feature_count=max_feature_id + 1,
            decoder_path=args.decoder_path,
            metadata_dir=metadata_dir,
            sae_release=args.sae_release,
            sae_name=args.sae_name,
            device=args.device,
        )
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Decoder weights contain zero-norm feature vectors")

    edges_kept = 0
    pf = pq.ParquetFile(args.coactivations_path)
    for batch in pf.iter_batches(columns=["feature_i", "feature_j", "jaccard"], batch_size=args.batch_size):
        jacc = batch.column("jaccard").to_numpy(zero_copy_only=False)
        mask = jacc >= args.jaccard_threshold
        if not mask.any():
            continue
        fi = batch.column("feature_i").to_numpy(zero_copy_only=False)[mask].astype(np.int32, copy=False)
        fj = batch.column("feature_j").to_numpy(zero_copy_only=False)[mask].astype(np.int32, copy=False)
        for a, b in zip(fi, fj):
            if a >= active_mask.size or b >= active_mask.size:
                continue
            if not active_mask[a] or not active_mask[b]:
                continue
            if cosine_threshold is not None and vectors is not None and norms is not None:
                denom = norms[a] * norms[b]
                if denom == 0:
                    continue
                cos = float(np.dot(vectors[a], vectors[b]) / denom)
                if cos > cosine_threshold:
                    continue
            uf.union(int(a), int(b))
            edges_kept += 1

    if not active_features:
        print("No active features after trimming")
        return

    component_sizes = {}
    for fid in active_features:
        root = uf.find(fid)
        component_sizes[root] = component_sizes.get(root, 0) + 1

    singleton_components = sum(1 for size in component_sizes.values() if size == 1)
    multi_sizes = [size for size in component_sizes.values() if size > 1]
    thresholds = [2, 3, 5, 10]
    counts_by_threshold = {t: sum(1 for size in multi_sizes if size > t) for t in thresholds}
    num_components = len(multi_sizes)
    largest = max(multi_sizes) if multi_sizes else 0
    features_in_multis = int(sum(multi_sizes))

    print(f"Jaccard threshold: {args.jaccard_threshold}")
    if cosine_threshold is not None:
        print(f"Cosine threshold: {cosine_threshold}")
    if density_threshold is not None:
        print(f"Density threshold: {density_threshold}")
        print(f"Features removed for density: {removed_for_density}")
    print(f"Active features: {len(active_features)}")
    print(f"Edges kept: {edges_kept}")
    print(f"Singleton components: {singleton_components}")
    print(f"Components (size>1): {num_components}")
    for t in thresholds:
        print(f"Components (size>{t}): {counts_by_threshold[t]}")
    print(f"Largest component size: {largest}")
    print(f"Features in multi-component clusters: {features_in_multis}")


if __name__ == "__main__":
    main()
