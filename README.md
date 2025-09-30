# Coactivation Manifolds Workflow Cheatsheet

All commands assume you are in the project root (`/workspace/coactivation_manifolds`) with the project virtual environment activated. When invoking the Python entrypoints directly, prefix with `PYTHONPATH=src` so the local package is importable.

```bash
# activate the project environment
source .venv/bin/activate
```

## 0. Generate Activations
Logs model + SAE activations into `activations/` and builds the feature index.

```bash
PYTHONPATH=src python scripts/0_generate_activations.py \
  data/test_acts \
  monology/pile-uncopyrighted train \
  --max-tokens 50000 \
  --stream-dataset \
  --device cuda
```

Key outputs: `data/test_acts/activations/`, `data/test_acts/metadata/feature_index.parquet`.

## 1. Compute Coactivations
Accumulates per-feature totals and pairwise Jaccard stats with a shard progress bar.

```bash
PYTHONPATH=src python scripts/1_compute_coactivations.py data/test_acts \
  --first-token-idx 100 \
  --output-name coactivations.parquet \
  --feature-counts-name feature_counts_trimmed.parquet
```

Outputs live in `data/test_acts/metadata/`.

Defaults process token positions `[0, 1024)`. Adjust `--first-token-idx` and
`--last-token-idx` to clamp the per-document window (e.g., `1` and `10`
capture the nine tokens immediately after BOS).

The PCA utilities (`4_plot_component_pca.py`, `visualize_3D_component_plots.py`)
inherit the same bounds from the trimmed feature counts by default; override
with their `--first-token-idx` / `--last-token-idx` flags when experimenting.

## 2. Plot Jaccard Percentiles
Visualises the tail of the Jaccard distribution on a log–log scale.

```bash
PYTHONPATH=src python scripts/2_plot_jaccard_percentiles.py \
  data/test_acts/metadata/coactivations.parquet \
  --output coactivations_percentiles.png \
  --min_tail 1e-4 --max_tail 1.0 --points 400
```

## 3. Prune Components
Builds the coactivation graph, applies thresholds, and reports component counts.

```bash
PYTHONPATH=src python scripts/3_prune_components.py \
  data/test_acts/metadata/coactivations.parquet \
  data/test_acts/metadata/feature_counts_trimmed.parquet \
  --jaccard-threshold 0.8 \
  --density-threshold 0.1 \
  --cosine-threshold 0.4 \
  --decoder-path data/test_acts/metadata/decoder_directions.npy
```

If `--decoder-path` is omitted the script caches decoder directions in `metadata/` automatically.

## 4. Plot Component PCA Grids
Streams activations once, runs PCA per component, and writes grid PNGs with feature IDs listed beside each plot.

```bash
PYTHONPATH=src python scripts/4_plot_component_pca.py data/test_acts \
  --jaccard-threshold 0.8 \
  --density-threshold 0.1 \
  --min-activations 64 \
  --max-pc 6 \
  --grid-size 10 \
  --output-dir component_pca_plots
```

Look for `component_pca_plots/component_pca_000.png`, etc., to review clusters quickly. Use `--no-progress` on any script to silence tqdm bars when desired.

## 5. Probability Simplex Experiments
Given a feature list, extract co-firing snippets, score definition choices with
Gemma-2-2b-it, and visualize the probability simplex in feature space.

```bash
# (optional) ensure token_text snippets exist on the run
PYTHONPATH=src python scripts/add_token_snippets.py data/test_acts

# 1) collect snippets where all features fire together
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.collect_joint_snippets \
  data/test_acts --features 12345 23456 34567 --max-examples 200

# 2) score the snippets against multiple-choice definitions
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.score_definitions \
  data/test_acts/metadata/probability_simplex/snippets_12345_23456_34567.parquet \
  --choice A="definition text for A" --choice B="definition text for B" --choice C="definition text for C"

# 3) project activations onto their top PCs and color by definition probabilities
PYTHONPATH=src python -m coactivation_manifolds.visualize_probability_simplex.plot_probability_simplex \
  data/test_acts/metadata/probability_simplex/snippets_12345_23456_34567_scored.parquet
```
