"""
Causal intervention experiment: steer model generations using probe-derived directions.

For ambiguous prefixes (multiple POS in ground truth), steer toward the minority class
and measure if the POS distribution shifts toward that class.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import jsonlines
import spacy

# Configuration
MODEL_NAME = "google/gemma-2-2b"
TARGET_LAYER = 6  # actual model layer to steer
STEERING_SCALE = 3000.0  # multiplier for steering vector
NUM_SAMPLES = 10  # completions per prefix
MAX_NEW_TOKENS = 12
TEMPERATURE = 1.0
TOP_P = 0.995
NUM_VAL_EXAMPLES = 24  # how many ambiguous prefixes to test
SEED = 1234

# Paths
DATA_DIR = Path(__file__).parent
VAL_INPUT_PATH = DATA_DIR / "data" / "../../1000_merged.jsonl"
PROBE_DIR = DATA_DIR / "models" / "per_layer_probes"
PCA_CACHE_BASE = DATA_DIR / "activation_cache" / "distill_cache_train_pca"
TAG_NAMES_PATH = DATA_DIR / "models" / "distilled_probe_tag_names.json"
OUTPUT_PATH = DATA_DIR / "steering_results.json"

# Layer mapping: LAYERS[i] = actual layer number
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
SPACY_MODEL = "en_core_web_trf"


def get_layer_index(actual_layer: int) -> int:
    """Convert actual layer number to index in LAYERS list."""
    return LAYERS.index(actual_layer)


def load_probe(layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load probe weights and bias for a specific layer."""
    probe_path = PROBE_DIR / f"probe_layer_{layer}.pt"
    probe_data = torch.load(probe_path, map_location="cpu")
    W = probe_data["state_dict"]["weight"]  # (num_classes, pca_dim)
    b = probe_data["state_dict"]["bias"]    # (num_classes,)
    return W, b


def load_pca_components(layer_idx: int) -> np.ndarray:
    """Load PCA components for inverse transform."""
    pca_path = PCA_CACHE_BASE.with_name(f"{PCA_CACHE_BASE.name}_pca_layer{layer_idx}.npz")
    pca_data = np.load(pca_path)
    return pca_data["components"]  # (pca_dim, hidden_dim)


def load_tag_names() -> List[str]:
    """Load POS tag names."""
    with open(TAG_NAMES_PATH) as f:
        return json.load(f)


def compute_steering_direction(
    target_class_idx: int,
    probe_W: torch.Tensor,
    pca_components: np.ndarray,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute steering direction in raw activation space.

    1. Create 1-hot for target class
    2. Multiply by probe weights transpose to get PCA-space direction
    3. Map through PCA inverse to get raw activation space direction
    4. Scale
    """
    num_classes = probe_W.shape[0]

    # 1-hot target
    pos_weights = torch.zeros(num_classes)
    pos_weights[target_class_idx] = 1.0

    # Direction in PCA space: W.T @ pos_weights
    # W is (num_classes, pca_dim), so W.T @ pos_weights gives (pca_dim,)
    d_pca = probe_W.T @ pos_weights  # (pca_dim,)

    # Map to raw activation space via PCA inverse
    # components is (pca_dim, hidden_dim)
    d_raw = d_pca.numpy() @ pca_components  # (hidden_dim,)

    # Scale and convert to tensor
    steering_vector = torch.from_numpy(d_raw * scale).float()
    return steering_vector


def get_minority_class(gt_distribution: np.ndarray) -> int | None:
    """Return index of minority class (smallest nonzero prob)."""
    nonzero_mask = gt_distribution > 0
    if nonzero_mask.sum() <= 1:
        return None  # Not ambiguous, skip
    # Among nonzero entries, find the smallest
    probs = np.where(nonzero_mask, gt_distribution, np.inf)
    return int(np.argmin(probs))


def build_gt_distribution(completions: List[dict], tag_to_idx: dict) -> np.ndarray:
    """Build ground truth distribution from completions."""
    counts = np.zeros(len(tag_to_idx), dtype=np.float64)
    total = 0
    for comp in completions:
        pos = comp.get("spacy_pos")
        if pos and pos in tag_to_idx:
            counts[tag_to_idx[pos]] += 1
            total += 1
    if total > 0:
        counts /= total
    return counts


class SteeringHook:
    """Hook to add steering vector at a fixed position."""

    def __init__(self, steering_vector: torch.Tensor, prefix_last_idx: int):
        self.steering_vector = steering_vector
        self.prefix_last_idx = prefix_last_idx

    def __call__(self, module, input, output):
        hidden = output[0]  # (batch, seq_len, hidden_dim)
        seq_len = hidden.shape[1]
        # Only add steering if prefix position is in current sequence
        if seq_len > self.prefix_last_idx:
            hidden[:, self.prefix_last_idx, :] = (
                hidden[:, self.prefix_last_idx, :] + self.steering_vector.to(hidden.device)
            )
        return (hidden,) + output[1:]


def load_val_records(path: Path, tag_names: List[str]) -> List[dict]:
    """Load validation records and filter to ambiguous examples."""
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    records = []

    with jsonlines.open(path) as reader:
        for obj in reader:
            if "metadata" in obj:
                continue
            candidate = obj.get("candidate", {})
            completions = obj.get("completions", [])

            if not candidate.get("prefix_token_ids"):
                continue

            # Build GT distribution
            gt_dist = build_gt_distribution(completions, tag_to_idx)

            # Check if ambiguous
            minority_class = get_minority_class(gt_dist)
            if minority_class is None:
                continue

            records.append({
                "prefix_token_ids": candidate["prefix_token_ids"],
                "prefix_text": candidate.get("prefix_text", ""),
                "target_token_text": candidate.get("target_token_text", ""),
                "gt_distribution": gt_dist,
                "minority_class": minority_class,
                "minority_class_name": tag_names[minority_class],
                "minority_class_prob": float(gt_dist[minority_class]),
            })

    return records


def generate_steered_completions(
    model,
    tokenizer,
    prefix_ids: List[int],
    steering_hook: SteeringHook,
    hook_handle,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """Generate completions with steering hook active."""
    input_ids = torch.tensor([prefix_ids], device="cuda")

    completions = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[0, len(prefix_ids):]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text)

    return completions


def classify_completions_spacy(
    completions: List[str],
    prefix_text: str,
    nlp,
) -> List[str | None]:
    """Classify the first token of each completion using spacy."""
    results = []

    for comp_text in completions:
        if not comp_text.strip():
            results.append(None)
            continue

        # Process the completion text
        full_text = prefix_text + comp_text
        doc = nlp(full_text)

        # Find the token that starts after the prefix
        prefix_len = len(prefix_text)
        target_pos = None

        for token in doc:
            if token.idx >= prefix_len:
                target_pos = token.pos_
                break

        results.append(target_pos)

    return results


def compute_steered_distribution(
    pos_labels: List[str | None],
    tag_to_idx: dict,
) -> np.ndarray:
    """Compute distribution from classified POS labels."""
    counts = np.zeros(len(tag_to_idx), dtype=np.float64)
    total = 0
    for pos in pos_labels:
        if pos and pos in tag_to_idx:
            counts[tag_to_idx[pos]] += 1
            total += 1
    if total > 0:
        counts /= total
    return counts


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading tag names...")
    tag_names = load_tag_names()
    tag_to_idx = {t: i for i, t in enumerate(tag_names)}
    print(f"Tags: {tag_names}")

    print(f"\nLoading probe for layer {TARGET_LAYER}...")
    probe_W, probe_b = load_probe(TARGET_LAYER)
    print(f"Probe weights shape: {probe_W.shape}")

    print(f"\nLoading PCA components...")
    layer_idx = get_layer_index(TARGET_LAYER)
    pca_components = load_pca_components(layer_idx)
    print(f"PCA components shape: {pca_components.shape}")

    print(f"\nLoading validation records...")
    records = load_val_records(VAL_INPUT_PATH, tag_names)
    print(f"Found {len(records)} ambiguous examples")

    # Sample subset
    if len(records) > NUM_VAL_EXAMPLES:
        np.random.shuffle(records)
        records = records[:NUM_VAL_EXAMPLES]
    print(f"Using {len(records)} examples")

    print(f"\nLoading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    print(f"\nLoading spacy model {SPACY_MODEL}...")
    spacy.require_gpu()
    nlp = spacy.load(SPACY_MODEL)

    # Run steering experiment
    results = []

    print(f"\nRunning steering experiment with scale={STEERING_SCALE}...")
    for rec in tqdm(records, desc="Processing prefixes"):
        prefix_ids = rec["prefix_token_ids"]
        prefix_text = rec["prefix_text"]
        minority_class = rec["minority_class"]
        gt_dist = rec["gt_distribution"]

        # Compute steering direction for this example's minority class
        steering_vector = compute_steering_direction(
            minority_class, probe_W, pca_components, scale=STEERING_SCALE
        )

        # Create hook at last prefix position
        prefix_last_idx = len(prefix_ids) - 1
        hook = SteeringHook(steering_vector, prefix_last_idx)
        hook_handle = model.model.layers[TARGET_LAYER].register_forward_hook(hook)

        try:
            # Generate steered completions
            completions = generate_steered_completions(
                model, tokenizer, prefix_ids, hook, hook_handle,
                NUM_SAMPLES, MAX_NEW_TOKENS, TEMPERATURE, TOP_P
            )
        finally:
            hook_handle.remove()

        # Classify with spacy
        pos_labels = classify_completions_spacy(completions, prefix_text, nlp)

        # Compute steered distribution
        steered_dist = compute_steered_distribution(pos_labels, tag_to_idx)

        # Store result
        result = {
            "prefix_text": prefix_text,
            "target_token": rec["target_token_text"],
            "minority_class": rec["minority_class_name"],
            "gt_minority_prob": rec["minority_class_prob"],
            "steered_minority_prob": float(steered_dist[minority_class]),
            "gt_distribution": gt_dist.tolist(),
            "steered_distribution": steered_dist.tolist(),
            "completions": completions,
            "pos_labels": pos_labels,
        }
        results.append(result)

        # Print progress
        gt_prob = rec["minority_class_prob"]
        steered_prob = steered_dist[minority_class]
        delta = steered_prob - gt_prob
        print(f"  {rec['minority_class_name']}: GT={gt_prob:.2%} → Steered={steered_prob:.2%} (Δ={delta:+.2%})")
        # Show prefix and first completion for qualitative verification
        print(f"    Prefix: {prefix_text[-60:]!r}")
        print(f"    Completion: {completions[0]!r}")

    # Compute aggregate metrics
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    gt_probs = [r["gt_minority_prob"] for r in results]
    steered_probs = [r["steered_minority_prob"] for r in results]
    deltas = [s - g for s, g in zip(steered_probs, gt_probs)]

    success_count = sum(1 for d in deltas if d > 0)

    print(f"Mean GT minority prob:      {np.mean(gt_probs):.2%}")
    print(f"Mean steered minority prob: {np.mean(steered_probs):.2%}")
    print(f"Mean delta:                 {np.mean(deltas):+.2%}")
    print(f"Success rate (delta > 0):   {success_count}/{len(results)} ({success_count/len(results):.1%})")

    # Save results
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "target_layer": TARGET_LAYER,
            "steering_scale": STEERING_SCALE,
            "num_samples": NUM_SAMPLES,
            "num_examples": len(results),
        },
        "aggregate": {
            "mean_gt_minority_prob": float(np.mean(gt_probs)),
            "mean_steered_minority_prob": float(np.mean(steered_probs)),
            "mean_delta": float(np.mean(deltas)),
            "success_rate": success_count / len(results),
        },
        "results": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
