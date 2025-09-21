from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("pyarrow")

from coactivation_manifolds.activation_reader import (
    FeatureIndexBuilder,
    FeatureIndexConfig,
    ActivationReader,
)
from coactivation_manifolds.activation_writer import ActivationWriter, ActivationRecord


def make_record(token_index: int) -> ActivationRecord:
    return ActivationRecord(
        doc_id=0,
        token_index=token_index,
        position_in_doc=token_index,
        feature_ids=[token_index % 5, (token_index + 1) % 5],
        activations=[0.5, 0.2],
        token_text=f"token-{token_index}",
    )


def test_writer_reader_round_trip(tmp_path):
    activations_dir = tmp_path / "activations"
    writer = ActivationWriter(activations_dir, num_features=16, shard_size_tokens=4)
    writer.add_records(make_record(i) for i in range(10))
    writer.finalize()

    index_path = tmp_path / "metadata" / "feature_index.parquet"
    builder = FeatureIndexBuilder(
        FeatureIndexConfig(activations_dir, index_path, num_features=16)
    )
    builder.build()

    reader = ActivationReader(activations_dir, index_path)
    table = reader.load_cluster([1, 2])
    feature_lists = table.column("feature_ids").to_pylist()
    assert all({1, 2}.intersection(set(fl)) for fl in feature_lists)

    activations = table.column("activations").to_pylist()
    assert all(len(a) == len(f) for a, f in zip(activations, feature_lists))
