"""Index building and retrieval helpers for activation shards."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass
class FeatureIndexConfig:
    activations_dir: Path | str
    output_path: Path | str
    num_features: int


class FeatureIndexBuilder:
    """Builds a sparse feature-to-token index from activation shards."""

    def __init__(self, config: FeatureIndexConfig) -> None:
        self.config = config
        self.activations_dir = Path(config.activations_dir)
        self.output_path = Path(config.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self) -> None:
        entries: List[Tuple[int, int, int, int]] = []  # feature_id, shard, row_start, row_end
        feature_counts = np.zeros(self.config.num_features, dtype=np.int64)

        shard_paths = sorted(self.activations_dir.glob("shard=*"), key=lambda p: p.name)
        for shard_path in shard_paths:
            shard_id = int(shard_path.name.split("=")[1])
            file_path = shard_path / "data.parquet"
            table = pq.read_table(file_path, columns=["feature_ids"])

            runs = self._collect_runs(table["feature_ids"].to_pylist())
            for feature_id, row_ids in runs.items():
                if feature_id >= self.config.num_features or feature_id < 0:
                    raise ValueError(
                        f"feature_id {feature_id} outside configured range [0, {self.config.num_features})"
                    )
                feature_counts[feature_id] += len(row_ids)
                for start, end in _compress_runs(row_ids):
                    entries.append((feature_id, shard_id, start, end))

        if entries:
            entries.sort()
            table = pa.table(
                {
                    "feature_id": pa.array([e[0] for e in entries], type=pa.int32()),
                    "shard": pa.array([e[1] for e in entries], type=pa.int32()),
                    "row_start": pa.array([e[2] for e in entries], type=pa.int32()),
                    "row_end": pa.array([e[3] for e in entries], type=pa.int32()),
                }
            )
        else:
            table = pa.table(
                {
                    "feature_id": pa.array([], type=pa.int32()),
                    "shard": pa.array([], type=pa.int32()),
                    "row_start": pa.array([], type=pa.int32()),
                    "row_end": pa.array([], type=pa.int32()),
                }
            )
        pq.write_table(table, self.output_path, compression="zstd")

        offsets_path = self.output_path.parent / "feature_offsets.npy"
        offsets = np.concatenate(([0], np.cumsum(feature_counts, dtype=np.int64)))
        np.save(offsets_path, offsets)

    @staticmethod
    def _collect_runs(feature_lists: List[List[int]]) -> Dict[int, List[int]]:
        rows: Dict[int, List[int]] = defaultdict(list)
        for row_idx, features in enumerate(feature_lists):
            for fid in features:
                rows[int(fid)].append(row_idx)
        return rows


class ActivationReader:
    """Retrieve cluster slices from activation shards using a feature index."""

    def __init__(self, activations_dir: Path | str, feature_index_path: Path | str) -> None:
        self.activations_dir = Path(activations_dir)
        self.feature_index = pq.read_table(feature_index_path)
        self._index = self._group_index(self.feature_index)

    def iter_cluster_records(self, feature_ids: Sequence[int]) -> Iterable[pa.Table]:
        feature_set = set(int(fid) for fid in feature_ids)
        for feature_id in feature_set:
            for shard, row_start, row_end in self._index.get(feature_id, []):
                shard_table = pq.read_table(self._shard_path(shard))
                slice_table = shard_table.slice(row_start, row_end - row_start)
                filtered = self._filter_table(slice_table, feature_set)
                if filtered is not None and filtered.num_rows > 0:
                    yield filtered

    def load_cluster(self, feature_ids: Sequence[int]) -> pa.Table:
        tables = list(self.iter_cluster_records(feature_ids))
        if not tables:
            raise ValueError("No activations found for the requested feature IDs")
        merged = pa.concat_tables(tables)
        token_indices = merged.column("token_index").to_pylist()
        seen = set()
        keep: List[int] = []
        for idx, token_id in enumerate(token_indices):
            if token_id in seen:
                continue
            seen.add(token_id)
            keep.append(idx)
        if not keep:
            return merged.slice(0, 0)
        return merged.take(pa.array(keep, type=pa.int32()))

    def _group_index(self, table: pa.Table) -> Dict[int, List[Tuple[int, int, int]]]:
        grouped: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        if table.num_rows == 0:
            return grouped
        features = table.column("feature_id").to_pandas().astype(int)
        shards = table.column("shard").to_pandas().astype(int)
        starts = table.column("row_start").to_pandas().astype(int)
        ends = table.column("row_end").to_pandas().astype(int)
        for fid, shard, start, end in zip(features, shards, starts, ends):
            grouped[int(fid)].append((int(shard), int(start), int(end)))
        for fid in grouped:
            grouped[fid].sort()
        return grouped

    def _shard_path(self, shard: int) -> Path:
        return self.activations_dir / f"shard={shard:06d}" / "data.parquet"

    def _filter_table(self, table: pa.Table, feature_ids: set[int]) -> pa.Table | None:
        feat_col = table.column("feature_ids").to_pylist()
        act_col = table.column("activations").to_pylist()
        keep_indices: List[int] = []
        filtered_feats: List[List[int]] = []
        filtered_acts: List[List[float]] = []

        for idx, (feat_list, act_list) in enumerate(zip(feat_col, act_col)):
            pairs = [(f, a) for f, a in zip(feat_list, act_list) if f in feature_ids]
            if not pairs:
                continue
            fids, acts = zip(*pairs)
            keep_indices.append(idx)
            filtered_feats.append(list(fids))
            filtered_acts.append(list(acts))

        if not keep_indices:
            return None

        take = pa.array(keep_indices, type=pa.int32())
        subset = table.take(take)
        feat_array = pa.array(filtered_feats, type=pa.list_(pa.uint16()))
        act_array = pa.array(filtered_acts, type=pa.list_(pa.float16()))
        feat_idx = subset.schema.get_field_index("feature_ids")
        act_idx = subset.schema.get_field_index("activations")
        subset = subset.set_column(feat_idx, "feature_ids", feat_array)
        subset = subset.set_column(act_idx, "activations", act_array)
        return subset


def _compress_runs(row_ids: Sequence[int]) -> List[Tuple[int, int]]:
    if not row_ids:
        return []
    runs: List[Tuple[int, int]] = []
    start = row_ids[0]
    prev = start
    for current in row_ids[1:]:
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev + 1))
        start = current
        prev = current
    runs.append((start, prev + 1))
    return runs
