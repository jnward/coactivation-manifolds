"""Tools for streaming SAE activations to Parquet shards.

This module focuses on persistence; integration with model inference lives in
`activation_pipeline.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


TOKEN_SCHEMA = pa.schema(
    [
        ("doc_id", pa.int64()),
        ("token_index", pa.int64()),
        ("position_in_doc", pa.int32()),
        ("feature_ids", pa.list_(pa.uint16())),
        ("activations", pa.list_(pa.float16())),
        ("token_text", pa.large_string()),
    ]
)


@dataclass(slots=True)
class ActivationRecord:
    """Single token activation payload ready for persistence."""

    doc_id: int
    token_index: int
    position_in_doc: int
    feature_ids: Sequence[int]
    activations: Sequence[float]
    token_text: str

    def validate(self) -> None:
        if len(self.feature_ids) != len(self.activations):
            raise ValueError(
                "feature_ids and activations must have the same length; "
                f"got {len(self.feature_ids)} and {len(self.activations)}"
            )


class ActivationWriter:
    """Stream tokens into chunked Parquet shards with auxiliary stats."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        num_features: int,
        shard_size_tokens: int = 8_192,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.output_dir.parent / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.num_features = int(num_features)
        self.shard_size_tokens = shard_size_tokens

        self._feature_counts = np.zeros(self.num_features, dtype=np.uint64)
        self._shard_idx = 0
        self._tokens_in_shard = 0
        self._buffer: List[ActivationRecord] = []
        self._shard_stats: List[dict] = []
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_records(self, records: Iterable[ActivationRecord]) -> None:
        for record in records:
            record.validate()
            self._buffer.append(record)
            self._update_counters(record)
            if self._tokens_in_shard >= self.shard_size_tokens:
                self._flush_buffer()

    def finalize(self) -> None:
        if self._closed:
            return

        self._flush_buffer(force=True)
        self._write_feature_counts()
        self._write_shard_stats()
        self._closed = True

    def __enter__(self) -> "ActivationWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_counters(self, record: ActivationRecord) -> None:
        indices = np.asarray(record.feature_ids, dtype=np.int64)
        if indices.size:
            if (indices < 0).any() or (indices >= self.num_features).any():
                raise ValueError("feature index out of bounds for configured num_features")
            np.add.at(self._feature_counts, indices, 1)
        self._tokens_in_shard += 1

    def _flush_buffer(self, *, force: bool = False) -> None:
        if not self._buffer:
            return

        if self._tokens_in_shard >= self.shard_size_tokens or force:
            self._write_shard(self._buffer)
            self._buffer.clear()
            self._tokens_in_shard = 0
            self._shard_idx += 1

    def _write_shard(self, records: Sequence[ActivationRecord]) -> None:
        if not records:
            return

        table = self._records_to_table(records)
        shard_dir = self.output_dir / f"shard={self._shard_idx:06d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        file_path = shard_dir / "data.parquet"
        pq.write_table(
            table,
            file_path,
            compression="zstd",
            compression_level=5,
        )

        # Compute shard stats for metadata report.
        nonzeros = [len(r.feature_ids) for r in records]
        flat_acts = (
            np.concatenate([
                np.asarray(r.activations, dtype=np.float32) for r in records if r.activations
            ])
            if records
            else np.array([], dtype=np.float32)
        )
        shard_stat = {
            "shard": int(self._shard_idx),
            "token_count": int(len(records)),
            "mean_nonzeros": float(np.mean(nonzeros) if nonzeros else 0.0),
            "median_nonzeros": float(np.median(nonzeros) if nonzeros else 0.0),
            "max_nonzeros": int(max(nonzeros) if nonzeros else 0),
            "min_activation": float(np.min(flat_acts) if flat_acts.size else 0.0),
            "max_activation": float(np.max(flat_acts) if flat_acts.size else 0.0),
        }
        self._shard_stats.append(shard_stat)

    def _records_to_table(self, records: Sequence[ActivationRecord]) -> pa.Table:
        doc_id = [int(r.doc_id) for r in records]
        token_index = [int(r.token_index) for r in records]
        pos_in_doc = [int(r.position_in_doc) for r in records]
        feature_ids = [np.asarray(r.feature_ids, dtype=np.uint16).tolist() for r in records]
        activations = [np.asarray(r.activations, dtype=np.float16).tolist() for r in records]
        token_text = [r.token_text for r in records]

        arrays = {
            "doc_id": pa.array(doc_id, type=pa.int64()),
            "token_index": pa.array(token_index, type=pa.int64()),
            "position_in_doc": pa.array(pos_in_doc, type=pa.int32()),
            "feature_ids": pa.array(feature_ids, type=pa.list_(pa.uint16())),
            "activations": pa.array(activations, type=pa.list_(pa.float16())),
            "token_text": pa.array(token_text, type=pa.large_string()),
        }
        return pa.Table.from_pydict(arrays, schema=TOKEN_SCHEMA)

    def _write_feature_counts(self) -> None:
        path = self.metadata_dir / "feature_counts.parquet"
        table = pa.table({
            "feature_id": pa.array(np.arange(self.num_features), type=pa.int32()),
            "count": pa.array(self._feature_counts, type=pa.uint64()),
        })
        pq.write_table(table, path, compression="zstd")

    def _write_shard_stats(self) -> None:
        path = self.metadata_dir / "shard_stats.parquet"
        if not self._shard_stats:
            table = pa.table(
                {
                    "shard": pa.array([], type=pa.int32()),
                    "token_count": pa.array([], type=pa.int32()),
                    "mean_nonzeros": pa.array([], type=pa.float32()),
                    "median_nonzeros": pa.array([], type=pa.float32()),
                    "max_nonzeros": pa.array([], type=pa.int32()),
                    "min_activation": pa.array([], type=pa.float32()),
                    "max_activation": pa.array([], type=pa.float32()),
                }
            )
        else:
            table = pa.table(
                {
                    "shard": pa.array([stat["shard"] for stat in self._shard_stats], type=pa.int32()),
                    "token_count": pa.array([stat["token_count"] for stat in self._shard_stats], type=pa.int32()),
                    "mean_nonzeros": pa.array([stat["mean_nonzeros"] for stat in self._shard_stats], type=pa.float32()),
                    "median_nonzeros": pa.array([stat["median_nonzeros"] for stat in self._shard_stats], type=pa.float32()),
                    "max_nonzeros": pa.array([stat["max_nonzeros"] for stat in self._shard_stats], type=pa.int32()),
                    "min_activation": pa.array([stat["min_activation"] for stat in self._shard_stats], type=pa.float32()),
                    "max_activation": pa.array([stat["max_activation"] for stat in self._shard_stats], type=pa.float32()),
                }
            )
        pq.write_table(table, path, compression="zstd")
