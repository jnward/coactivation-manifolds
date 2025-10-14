# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository analyzes **geometric structures in sparse autoencoder (SAE) features** from language models by studying coactivation patterns. The core hypothesis: features that frequently fire together may form interpretable geometric structures (circles, simplices, helices) even when their weight vectors have low cosine similarity.

**Model Stack**: Gemma-2-2b/9b with GemmaScope SAEs (65k-131k features)
**Key Dependencies**: PyTorch, sae-lens, PyArrow/Parquet, transformers, sklearn

## Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# All scripts require PYTHONPATH=src
export PYTHONPATH=src

# HF_TOKEN must be set (via .env or environment) for model downloads
```

## Core Pipeline

The analysis follows a 5-stage pipeline:

### 0. Generate Activations
Stream tokens through Gemma+SAE and persist sparse activations to Parquet shards.

```bash
PYTHONPATH=src python scripts/0_generate_activations.py \
  data/test_acts \
  monology/pile-uncopyrighted train \
  --max-tokens 50000 \
  --stream-dataset \
  --device cuda
```

**Outputs**: `data/test_acts/activations/shard=XXXXXX/data.parquet`, `metadata/feature_index.parquet`, `metadata/feature_counts.parquet`

### 1. Compute Coactivations
Calculate pairwise Jaccard similarity and per-feature counts over a specified token window.

```bash
PYTHONPATH=src python scripts/1_compute_coactivations.py data/test_acts \
  --first-token-idx 0 \
  --last-token-idx 1024 \
  --output-name coactivations.parquet \
  --feature-counts-name feature_counts_trimmed.parquet
```

**Token Window**: `--first-token-idx` and `--last-token-idx` define per-document position bounds. Default `[0, 1024)` processes all positions. Use `[1, 10)` to exclude BOS and focus on early tokens.

**Outputs**: `metadata/coactivations.parquet` (feature pairs with Jaccard scores), `metadata/feature_counts_trimmed.parquet` (per-feature totals within window)

### 2. Plot Jaccard Percentiles
Visualize the tail of the Jaccard distribution to inform threshold selection.

```bash
PYTHONPATH=src python scripts/2_plot_jaccard_percentiles.py \
  data/test_acts/metadata/coactivations.parquet \
  --output coactivations_percentiles.png
```

### 3. Prune Components
Build coactivation graph, apply thresholds (Jaccard/cosine/density), and report connected components.

```bash
PYTHONPATH=src python scripts/3_prune_components.py \
  data/test_acts/metadata/coactivations.parquet \
  data/test_acts/metadata/feature_counts_trimmed.parquet \
  --jaccard-threshold 0.8 \
  --density-threshold 0.1 \
  --cosine-threshold 0.4
```

**Thresholds**:
- `--jaccard-threshold`: Minimum Jaccard similarity to keep edge (higher = stricter coactivation)
- `--density-threshold`: Remove features that fire on >X fraction of tokens (filters ubiquitous features)
- `--cosine-threshold`: Maximum cosine similarity between decoder directions (filters features that are nearly collinear; used to find features spanning high-dimensional concepts)

**Decoder Caching**: If `--decoder-path` omitted, script downloads SAE and caches decoder matrix to `metadata/decoder_directions.npy` automatically.

### 4. Plot Component PCA
For each component, stream activations, run PCA, and generate 2D grid plots of principal component pairs.

```bash
PYTHONPATH=src python scripts/4_plot_component_pca.py data/test_acts \
  --jaccard-threshold 0.8 \
  --density-threshold 0.1 \
  --min-activations 64 \
  --max-pc 6 \
  --grid-size 10 \
  --output-dir component_pca_plots
```

**Grid Structure**: Each PNG shows PC pairs (PC1 vs PC2, PC2 vs PC3, etc.) up to `--max-pc`, with feature IDs listed beside each subplot.

### 5. Probability Simplex (Experimental)
Collect co-firing snippets, score against multiple-choice definitions with Gemma-2-2b-it, and visualize probability simplex.

```bash
# Ensure token_text snippets exist
PYTHONPATH=src python scripts/add_token_snippets.py data/test_acts

# Collect snippets where specified features fire together
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.collect_joint_snippets \
  data/test_acts --features 12345 23456 34567 --max-examples 200

# Score snippets against definitions
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.score_definitions \
  data/test_acts/metadata/probability_simplex/snippets_12345_23456_34567.parquet \
  --choice A="definition A" --choice B="definition B" --choice C="definition C"

# Plot probability simplex
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.plot_probability_simplex \
  data/test_acts/metadata/probability_simplex/snippets_12345_23456_34567_scored.parquet
```

## Architecture

### Data Flow

```
Raw Text (HF Dataset)
  ↓ 0_generate_activations.py (activation_pipeline.py + activation_writer.py)
Parquet Shards (activations/shard=XXXXXX/data.parquet) + Feature Index
  ↓ 1_compute_coactivations.py (coactivation_stats.py)
Coactivation Matrix (metadata/coactivations.parquet)
  ↓ 3_prune_components.py (component_graph.py)
Connected Components (in-memory clusters)
  ↓ 4_plot_component_pca.py (activation_reader.py)
PCA Visualizations (component_pca_plots/)
```

### Key Modules

**`activation_writer.py`**: Streams `ActivationRecord` objects to Parquet shards (~8k tokens/shard). Tracks per-feature counts and shard stats. Schema: `doc_id`, `token_index`, `position_in_doc`, `feature_ids: list<uint16>`, `activations: list<float16>`, `token_text`.

**`activation_pipeline.py`**: Runs Gemma+SAE inference in batches, emits `ActivationRecord` objects. Handles tokenization, hidden state extraction (specified layer), SAE encoding, and ReLU activation. Produces token snippets with «center token» markers.

**`activation_reader.py`**: Builds feature-to-token index (`feature_index.parquet`) mapping each feature to `(shard, row_start, row_end)` slices. `ActivationReader.load_cluster(feature_ids)` retrieves all tokens where any cluster feature fires, filtering feature lists to the cluster subset and deduplicating by `token_index`.

**`coactivation_stats.py`**: Scans shards to accumulate per-feature counts and pairwise intersection counts. Computes Jaccard similarity: `intersection / (count_i + count_j - intersection)`. Respects `first_token_idx` / `last_token_idx` bounds via `position_in_doc` filtering.

**`component_graph.py`**: Union-find clustering over Jaccard-thresholded edges. Optional cosine similarity upper bound to filter near-collinear features. Optional density threshold to remove features that fire too frequently. Returns `ComponentGraphResult` with components, edge stats, and metadata.

**`sae_loader.py`**: Thin wrapper around sae-lens. Defaults to `gemma-scope-9b-pt-res-canonical` release, `layer_20/width_131k/canonical` SAE. Infers feature count via `d_sae` / `n_features` / `cfg.d_sae`.

### Storage Format

**Parquet Schema (activations)**:
- Sparse representation: each token stores only nonzero feature activations
- `feature_ids: list<uint16>` and `activations: list<float16>` have same length
- Partitioned by shard for parallel writes and efficient range scans
- ZSTD compression level 5 balances size and decode speed

**Feature Index**:
- Maps `feature_id → [(shard, row_start, row_end)]` to avoid full shard scans
- Built once after activation logging completes
- Cached `feature_offsets.npy` stores cumulative counts for quick slicing

### Token Window Semantics

**`position_in_doc`**: Zero-indexed position within each document (resets per document).

**`first_token_idx` / `last_token_idx`**: Half-open interval `[first, last)` applied to `position_in_doc` during coactivation computation. Scripts 1, 4, and probability simplex utilities inherit these bounds from `feature_counts_trimmed.parquet` metadata unless overridden.

**Use Case**: Set `[1, 10)` to exclude BOS and analyze only the first 9 content tokens per document.

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_activation_io.py -v
```

**`tests/test_activation_io.py`**: Smoke test for writer → index builder → reader round-trip with synthetic data.

## Common Patterns

**Changing SAE**: Pass `--sae-release` and `--sae-name` to scripts 0 and 3. Ensure feature counts match across pipeline stages.

**Inspecting Metadata**: All Parquet files are readable via `pyarrow.parquet.read_table()` or pandas. Example:
```python
import pyarrow.parquet as pq
table = pq.read_table("metadata/coactivations.parquet")
table.schema.metadata  # Check for token_count, first_token_idx, last_token_idx
```

**Decoder Caching**: Script 3 auto-downloads and caches decoder weights to `metadata/decoder_directions.npy` on first run with `--cosine-threshold`. Reuse across runs by omitting `--decoder-path`.

**Progress Bars**: Most scripts use tqdm. Add `--no-progress` (if supported) to suppress bars in batch jobs.

## Design References

See `DESIGN.md` for the full research plan including activation persistence rationale, indexing strategy, and expected geometric structures.

See `README.md` for quick-reference command templates and workflow overview.
