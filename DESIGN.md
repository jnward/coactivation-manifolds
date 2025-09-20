# Discovering Multi-Dimensional Feature Geometry Through Coactivation Statistics

## Background

Engels et al. (2024) discovered interpretable geometric structures in language models (circles for days of week, months of year) by clustering sparse autoencoder (SAE) dictionary elements based on cosine similarity of their weight vectors. They found that when multiple SAE features represent the same multi-dimensional concept, they cluster together with high cosine similarity.

However, this weight-based approach may miss important structures. Features that frequently co-activate might form meaningful geometric patterns even when their weight vectors are dissimilar or orthogonal. For example, belief state representations (simplices) might have SAE features with zero or negative cosine similarity that nonetheless co-activate in structured ways.

## Objective

Find clusters of SAE features that:
1. Frequently co-activate (high Jaccard similarity) 
2. Form interpretable geometric structures (circles, simplices, helices) when analyzed via PCA
3. Would potentially be missed by cosine similarity clustering

## Method

### Setup
- **Model**: Gemma-2-2b (base model, not instruct)
- **SAE**: GemmaScope 65k features at layer 12 (middle layer balances abstraction with interpretability)
- **Dataset**: Large diverse text corpus

### Coactivation Analysis

1. For each token in dataset, get SAE activations (post-ReLU)
2. Create binary activation matrix A of shape [n_tokens, n_features] where A[i,j] = 1 if feature j fires on token i
3. Compute Jaccard similarity for all feature pairs:
   ```
   intersection = tokens where both features fire
   union = tokens where either feature fires
   jaccard[i, j] = |intersection| / |union|
   ```
4. Also compute cosine similarity matrix from SAE decoder weights D for comparison

### Clustering

1. Build graph G with SAE features as nodes
2. Add edges between features based on Jaccard similarity
3. Prune edges below threshold T (clusters are connected components)
4. Optional experiment: Also prune edges where cosine similarity is very high (>0.8) to attempt to filter for features spanning basis dimensions for high-D features.

The threshold T controls cluster granularity - we want clusters of ~2-50 features.

### Geometric Analysis

For each cluster:

1. **Collect activations**: For all tokens where ANY feature in the cluster activates, extract the activation vector restricted to just the cluster features
2. **PCA projection**: Run PCA on these activation vectors
3. **Visualize**: Plot 2D projections (PC1 vs PC2, PC2 vs PC3, PC3 vs PC4, PC4 vs PC5)
4. **Interpret**: Color points by current token, next token, or position to look for patterns

We expect to potentially find:
- Circles/cycles (like Engels' days of week)
- Simplices (belief states)
- Other geometric structures

### Comparison

For clusters with interesting geometry, check:
- Would these features cluster under cosine similarity?
- What tokens activate each feature?
- Do the geometric structures make semantic sense?

## Challenges

### Technical
- Efficient computation of coactivation statistics (65k × 65k matrix)
- Efficient computation of cluster activations for dataset examples with nonzero activations

### Methodological
- Finding threshold values for Jaccard similarity and cosine similarity
  - Can tune experimentally based on number and size of resulting clusters

## Activation Persistence Plan

### Goals
- Persist a single activation pass over ~1M tokens while keeping per-query reads memory-light
- Support repeated clustering experiments without recomputing model forward passes
- Make cluster-specific slices fast to materialize from disk-resident data

### On-Disk Layout
- Store activations as a PyArrow Parquet dataset under `activations/`, partitioned by shard index (e.g., `shard=000000`).
- Each shard covers ~8k tokens (tunable) so we stream-write without large resident buffers.
- Row schema per token:
  - `doc_id: int64` (source document identifier)
  - `token_index: int64` (position within corpus)
  - `position_in_doc: int32`
  - `feature_ids: list<uint16>` (sorted unique SAE feature indices)
  - `activations: list<float16>` (same length as `feature_ids`)
- Encode list columns as Arrow large list with `run_length_bitpacking` and Parquet ZSTD compression level 5 to balance size and decode speed.

- Add a streaming hook around the Gemma+SAE forward loop that accumulates tokens into an Arrow `RecordBatch` of size `batch_tokens` (target 512–1024) before writing to the current shard file.
- Track per-feature activation counts alongside shard-level statistics so we can estimate sparsity before clustering.
- Rotate to a new shard file when the shard reaches ~8k tokens or 2 GB, whichever comes first.
- After each shard flush, persist auxiliary summaries to `metadata/`:
  - `shard_stats.parquet`: tokens per shard, nonzeros per token, min/max activation magnitude
  - `feature_counts.parquet`: per-feature activation counts (dense `uint32` array)

### Indexing for Fast Reads
- Build a feature-to-token sparse index after the logging pass:
  - Scan shards sequentially and write `feature_index.parquet` with rows (`feature_id`, `shard`, `row_start`, `row_end`).
  - Offsets store the slice of token rows in each shard where the feature appears, using cumulative counts gathered during the scan.
- Cache `feature_offsets.npy` (float32 cumulative sums) for quick slicing during analysis.

### Read Path
- Provide a reader helper that, given a list of feature IDs, resolves the relevant `(shard, row range)` pairs from `feature_index.parquet` and streams only those tokens.
- When materializing a cluster, load the corresponding rows lazily and filter feature lists to the cluster IDs in-place to minimize deserialization time.
- For PCA prep, optionally spill retrieved activations to an Arrow table and hand off to NumPy/PyTorch once the slice fits in memory (<2M entries expected for cluster sizes up to 50).

### Implementation Milestones
1. Prototype activation writer on ~50k real tokens sampled from the target corpus; validate shard rollover, schema, and size (~150–300 GB extrapolation) using Gemma-2-2b with the production SAE.
2. Implement index builder pass and round-trip the read helper against the same logged subset to confirm filtering logic.
3. Integrate reader API into clustering pipeline and benchmark cluster extraction latency.

### Downstream Analysis Workflow
- Compute dense Jaccard similarity and cosine similarity matrices across SAE features using the persisted activation log and decoder weights.
- Construct the SAE feature graph with all-to-all edges and run pruning sweeps over Jaccard thresholds (and optional cosine caps), tracking number of clusters and median cluster size to locate elbow points.
- Extract token activations for each candidate cluster directly from disk into the cluster-restricted feature basis.
- Run PCA on the gathered cluster activations and cache principal components for comparison runs.
- Plot low-dimensional projections (PC1–PC2, etc.) with token-level annotations to visually scan for geometric structures.

### Implementation Artifacts
- `coactivation_manifolds/activation_writer.py`: streaming Parquet shard writer with feature counts and shard stats.
- `coactivation_manifolds/activation_pipeline.py`: Gemma+SAE runner that feeds the writer from a Hugging Face dataset.
- `coactivation_manifolds/activation_reader.py`: feature index builder plus cluster-oriented slice loader.
- `coactivation_manifolds/sae_loader.py`: sae-lens helper that fetches the GemmaScope SAE and exposes metadata like feature count.
- `scripts/run_logging.py`: CLI that ties everything together (model/tokenizer/SAE/dataset → shards + feature index).

## Expected Outcomes

If successful, we should find geometric structures formed by features that co-activate but may have low or negative cosine similarity. This would demonstrate that functional relationships (coactivation) can reveal different geometric organization than weight-based similarity.
