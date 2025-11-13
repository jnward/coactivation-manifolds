# Parts-of-Speech Probing

This project trains linear part-of-speech (POS) probes on the Gemma-2-2b
language model. It lives inside the main `coactivation-manifolds` repository so
it can share the existing Poetry environment.

## Capabilities

- Download and preprocess the Universal Dependencies (English-EWT) dataset.
- Align POS tags to the final subtoken of each word (causal-model friendly).
- Extract Gemma residual-stream activations at a configurable layer.
- Train an L2-regularized linear probe (PyTorch) on cached activations.
- Evaluate with either softmax probabilities or a ReLU-normalize variant.
- Inspect predictions on sample sentences and save learned weights for reuse.

## Layout

```
parts-of-speech/
├── README.md
├── train_probe.py
├── inspect_probe.py
└── pos_probe/
    ├── __init__.py
    ├── activations.py
    ├── constants.py
    ├── data.py
    ├── probe.py
    └── utils.py
```

All Python modules live under `pos_probe/`. The top-level scripts add this
directory to `sys.path`, so you can run them directly via:

```bash
poetry run python parts-of-speech/train_probe.py --help
```

## Usage

### Train + Evaluate a Probe

```bash
poetry run python parts-of-speech/train_probe.py \
  --model-name google/gemma-2-2b \
  --layer-index 12 \
  --extract-batch-size 8 \
  --probe-batch-size 1024 \
  --epochs 5 \
  --normalize-activations
```

This command:

- Downloads & tokenizes UD English-EWT (train/validation/test).
- Extracts residual activations at layer 12 (cached under `parts-of-speech/cache/`).
- Trains an L2-regularized linear probe on cached activations.
- Reports validation loss/accuracy per epoch, final test loss/accuracy, and per-tag recall (for both probability modes), then writes:
  - Weights to `parts-of-speech/runs/<model>/layer_<n>/probe.pt`
  - Metrics/history to `parts-of-speech/runs/<model>/layer_<n>/metrics.json`
  - Confusion matrix plots (counts + normalized) to
    `parts-of-speech/runs/<model>/layer_<n>/confusion_matrix_softmax*.png`

Re-run with `--recompute-activations` to refresh caches or change the layer index/model name.

#### Use the sklearn Logistic Regression Solver

```bash
poetry run python parts-of-speech/train_probe.py \
  --trainer sklearn \
  --sklearn-max-iter 500 \
  --sklearn-C 0.5
```

This fits a multinomial logistic regression via scikit-learn on the cached activations, converts the learned weights back into the PyTorch probe module, and produces the same metrics/weight artifacts for side-by-side comparisons.

#### Train One-vs-Rest Probes

```bash
poetry run python parts-of-speech/train_probe.py \
  --trainer sklearn \
  --classification-mode ovr \
  --sklearn-class-weight balanced
```

This trains 17 binary classifiers (one per tag) with balanced class weights, stacks their logits into a single probe, and evaluates it identically to the multiclass setup. Useful when the multinomial probe collapses to the majority class.

#### Drop Specific Tags (e.g., INTJ)

```bash
poetry run python parts-of-speech/train_probe.py \
  --trainer sklearn \
  --classification-mode ovr \
  --disabled-tags INTJ
```

Any tag listed in `--disabled-tags` is removed from the training data, probe head, metrics, and confusion matrices (stored as `active_tags` in the metadata). This is handy when extremely rare tags destabilize the probe.

### Plot Confusion Matrix Without Specific Tags

You can re-render a stored confusion matrix after masking problematic tags (e.g., INTJ) without retraining:

```bash
poetry run python parts-of-speech/plot_confusion_subset.py \
  --metrics parts-of-speech/runs/google-gemma-2-2b/layer_12/metrics.json \
  --drop-tags INTJ \
  --normalize true
```

The script reads the confusion counts from `metrics.json`, filters rows/columns matching the dropped tags, and writes a new PNG (default `<metrics_stem>_subset.png`).

### Inspect Predictions

After training, load the weights and inspect predictions on a handful of test sentences:

```bash
poetry run python parts-of-speech/inspect_probe.py \
  --weights parts-of-speech/runs/google-gemma-2-2b/layer_12/probe.pt \
  --layer-index 12 \
  --num-sentences 5 \
  --drop-tags INTJ
```

This prints each word, gold tag, and predicted tag (for both softmax and ReLU-normalized probabilities by default). The optional `--drop-tags` parameter masks those tags at inference time so they never win the argmax and are labelled “dropped” if encountered.

## Next Steps

1. Implement experiment configs + logging helpers (e.g., wandb or tensorboard).
2. Add optional scikit-learn logistic regression training for comparison.
3. Support probing multiple layers in a single run and plotting accuracy vs. layer index.
