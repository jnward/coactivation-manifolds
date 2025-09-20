# Progress Log

1. Activation persistence implemented (`src/coactivation_manifolds/activation_writer.py`; shards + stats).
2. Gemma+SAE streaming pipeline emits Parquet records (`src/coactivation_manifolds/activation_pipeline.py`).
3. Feature index and reader for cluster extraction (`src/coactivation_manifolds/activation_reader.py`).
4. Minimal sae-lens loader plus CLI with Gemma/GemmaScope defaults (`src/coactivation_manifolds/sae_loader.py`, `scripts/0_generate_activations.py`).
5. Dotenv integration and built-in tqdm progress reporting in the pipeline.
6. Smoke test (`tests/test_activation_io.py`) validates writer→index→reader.

## Next Steps
- Install dependencies locally (`pip install -e .`) and ensure `python-dotenv` is available so `HF_TOKEN` loads automatically.
- Run `scripts/0_generate_activations.py /path/to/output --max-tokens <N>` to generate activation shards and the feature index.
- Inspect `metadata/feature_counts.parquet` and `metadata/shard_stats.parquet` to gauge sparsity before computing Jaccard similarities.
