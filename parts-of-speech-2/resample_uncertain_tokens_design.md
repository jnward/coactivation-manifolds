# Resampling Uncertain POS Tokens

Plan for generating on-policy continuations from Gemma 2 2B for validation tokens where the trained probe is uncertain, so we can later approximate true POS distributions conditioned on prefixes.

## Objectives
- Prove probe uncertainty reflects genuine ambiguity by sampling Gemma continuations and labeling the target token’s POS from completed sentences.
- Keep everything reproducible: fixed configs, deterministic filtering, saved metadata for later human/automatic labeling.
- Save raw generations (prefix + completions) so downstream scripts can attach POS labels without re-running LLM sampling.

## Inputs & Dependencies
1. **Probe + label logic**  
   - Load `probe.joblib` (or configurable path).  
   - Map excluded labels exactly as in `train_probes.py` (user mentioned `train-probes.pot`, referring to same script). Keep `_`, `SYM`, `INTJ` excluded by default, using `tag_name_to_upos`.
2. **Validation activations / dataset**  
   - Reuse `universal_dependencies/en_ewt` validation split.  
   - Tokenization + label alignment via `tokenize_with_labels` to stay consistent with training.
3. **Model assets**  
   - `google/gemma-2-2b` loaded in bf16 on a single GPU.

## Configuration Constants
```python
PROBE_PATH = "probe.joblib"
MODEL_NAME = "google/gemma-2-2b"
VAL_SPLIT = "validation"
TARGET_LAYER = 12        # example; adjustable
CONF_THRESH = 0.95
NUM_SAMPLES = 96
BATCH_SIZE = 96          # logical per-prefix batch; lower if OOM
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.7        # “canonical” default; allow override
TOP_P = 0.95             # optional; can be None
SEED = 123
OUTPUT_PATH = "uncertain_resamples.jsonl"
MAX_PREFIXES = None      # optional debug limiter
```
- Batch size interpretation: process one prefix at a time conceptually, but under the hood replicate the prefix `NUM_SAMPLES` times (so tensor batch = 96). If memory is tight, split into chunks (e.g., 32 + 32 + 32).

## Pipeline
1. **Preload assets & configs**
   - Initialize RNGs.
   - Load tokenizer/model (`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`).
   - Load probe with `joblib`, confirm `clf.classes_` ordering.

2. **Scan validation set for uncertain tokens**
   - Iterate deterministically (seed shuffle or plain order).
   - For each sentence, tokenize + align labels.
   - Run Gemma forward pass once (with `output_hidden_states=True`), grab `hidden_states[TARGET_LAYER]`.
   - For each position:
     - Skip unlabeled tokens or excluded labels.
     - Get `probs = clf.predict_proba(hidden_state[None])[0]`.
     - If `probs.max() < CONF_THRESH`, record candidate:
       - Metadata: dataset index, raw text, model `input_ids` up to current token (inclusive), `tokenizer.decode` for prefix text, target token id/text, true label id (using same mapping as training), probe probs, argmax tag, token position, context offsets if useful.
   - Optionally stop once `MAX_PREFIXES` reached.

3. **Generation stage**
   - For each candidate prefix:
     - Build tensor of prefix ids (single example) and replicate via `repeat_interleave(NUM_SAMPLES, dim=0)`.
     - Call `model.generate(... do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id)`.
     - After generation, reshape outputs back to `[NUM_SAMPLES, seq_len]`, slice off the prefix (start from original length) to isolate completions.
     - Decode completions to text; also keep token IDs for later tagger runs.
     - Record generation metadata (seed offset, sampling params, actual `num_return_sequences`).
   - If GPU OOM occurs, automatically split `NUM_SAMPLES` into sub-batches (e.g., 32) per prefix and concatenate the resulting completions list before saving.

4. **Persistence**
   - Append one JSON object per prefix to `OUTPUT_PATH` (uncompressed for now). Proposed schema:
     ```json
     {
       "dataset_idx": 1234,
       "prefix": {
         "text": "... prefix up to target ...",
         "token_ids": [...],
         "target_position": 17,
         "target_token": "lead",
         "true_label": 6,
         "probe_probs": [0.02, 0.11, ...],
         "probe_pred": "NOUN"
       },
       "sentence_text": "Full original sentence ...",
       "generation_params": {
         "model": "google/gemma-2-2b",
         "layer": 12,
         "temperature": 0.7,
         "top_p": 0.95,
         "max_new_tokens": 16,
         "num_samples": 96,
         "seed": 123
       },
       "completions": [
         {"text": " completion 1 ...", "token_ids": [...]},
         ...
       ]
     }
     ```
   - Include a short header record (first line) or companion metadata file describing global settings.

## Logging & Guardrails
- tqdm progress for both scanning and generation.
- Warn if no tokens meet the confidence filter.
- Track average completion length; ensure EOS handling.
- Validate that `probe_probs` sums to 1 and that `true_label` is within `clf.classes_`.
- Save periodic checkpoints (e.g., flush every prefix) to protect against crashes.

## Future Extensions
- **POS labeling**: run a POS tagger (or human labeling) on stored completions, focusing only on the target token in each context.
- **Multi-GPU scaling**: shard candidate list and run the script with shard IDs; concatenate JSON later.
- **Compression**: switch to `.jsonl.zst` or `.npz` once schema stabilizes.
- **Analysis scripts**: compute empirical POS distributions vs. probe predictions, calibration curves, etc.

