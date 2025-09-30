"""Shared helpers for building coactivation graph components."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from .sae_loader import DEFAULT_SAE_NAME, DEFAULT_SAE_RELEASE, load_sae


@dataclass
class ComponentGraphConfig:
    coactivations_path: Path
    feature_counts_path: Path
    jaccard_threshold: float = 0.8
    cosine_threshold: Optional[float] = None
    decoder_path: Optional[Path] = None
    sae_release: str = DEFAULT_SAE_RELEASE
    sae_name: str = DEFAULT_SAE_NAME
    device: str = "cpu"
    density_threshold: Optional[float] = None
    batch_size: int = 1_000_000


@dataclass
class ComponentGraphResult:
    components: List[List[int]]
    active_features: set[int]
    feature_counts: np.ndarray
    token_count: int
    first_token_idx: int
    last_token_idx: int
    edges_kept: int
    singleton_components: int
    removed_for_density: int


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
    config: ComponentGraphConfig,
    metadata_dir: Path,
) -> np.ndarray:
    target_path = (
        config.decoder_path
        if config.decoder_path is not None
        else metadata_dir / "decoder_directions.npy"
    )
    target_path = Path(target_path)
    if target_path.exists():
        vectors = _load_array(target_path)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        handle = load_sae(
            sae_release=config.sae_release,
            sae_name=config.sae_name,
            device=config.device,
        )
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
    return vectors.astype(np.float32, copy=False)


def compute_components(config: ComponentGraphConfig) -> ComponentGraphResult:
    counts_table = pq.read_table(config.feature_counts_path, columns=["feature_id", "count"])
    metadata = counts_table.schema.metadata or {}
    token_count_raw = metadata.get(b"token_count")
    if token_count_raw is None:
        raise ValueError(
            "feature_counts_trimmed.parquet missing token_count metadata; rerun 1_compute_coactivations.py"
        )
    token_count = int(token_count_raw.decode())
    first_token_idx = int(metadata.get(b"first_token_idx", b"0").decode())
    last_token_idx = int(metadata.get(b"last_token_idx", b"-1").decode())

    feature_ids = counts_table.column("feature_id").to_numpy(zero_copy_only=False)
    counts = counts_table.column("count").to_numpy(zero_copy_only=False)

    if len(feature_ids) == 0:
        return ComponentGraphResult(
            components=[],
            active_features=set(),
            feature_counts=counts,
            token_count=token_count,
            first_token_idx=first_token_idx,
            last_token_idx=last_token_idx,
            edges_kept=0,
            singleton_components=0,
            removed_for_density=0,
        )

    max_feature_id = int(feature_ids.max())
    uf = UnionFind(max_feature_id + 1)

    active_mask = np.zeros(max_feature_id + 1, dtype=bool)
    density_threshold = config.density_threshold
    removed_for_density = 0
    for fid, count in zip(feature_ids, counts):
        idx = int(fid)
        if count <= 0 or idx >= active_mask.size:
            continue
        if density_threshold is not None and token_count > 0:
            density = float(count) / float(token_count)
            if density > density_threshold:
                removed_for_density += 1
                continue
        active_mask[idx] = True

    active_indices = np.nonzero(active_mask)[0]
    active_features = set(int(i) for i in active_indices)

    cosine_threshold = config.cosine_threshold
    vectors = None
    norms = None
    if cosine_threshold is not None:
        metadata_dir = Path(config.feature_counts_path).resolve().parent
        vectors = _resolve_decoder_vectors(
            feature_count=max_feature_id + 1,
            config=config,
            metadata_dir=metadata_dir,
        )
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Decoder weights contain zero-norm feature vectors")

    edges_kept = 0
    pf = pq.ParquetFile(config.coactivations_path)
    batch_iter = pf.iter_batches(
        columns=["feature_i", "feature_j", "jaccard"],
        batch_size=config.batch_size,
    )
    for batch in tqdm(batch_iter, desc="Edges", leave=False):
        jacc = batch.column("jaccard").to_numpy(zero_copy_only=False)
        mask = jacc >= config.jaccard_threshold
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

    component_map: Dict[int, List[int]] = {}
    for fid in active_features:
        root = uf.find(fid)
        component_map.setdefault(root, []).append(fid)

    components = [sorted(features) for features in component_map.values()]
    singleton_components = sum(1 for feats in components if len(feats) == 1)

    return ComponentGraphResult(
        components=components,
        active_features=active_features,
        feature_counts=counts,
        token_count=token_count,
        first_token_idx=first_token_idx,
        last_token_idx=last_token_idx,
        edges_kept=edges_kept,
        singleton_components=singleton_components,
        removed_for_density=removed_for_density,
    )
