import json
import random
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import tokenize_with_labels, upos_to_tag_name, tag_name_to_upos


PROBE_PATH = "probe.joblib"
MODEL_NAME = "google/gemma-2-2b"
SPLIT = "validation"
TARGET_LAYER = 12
CONF_THRESH = 0.95
NUM_SAMPLES = 100
CHUNK_SIZE = 100  # how many samples to draw per generation call
MAX_NEW_TOKENS = 32
TEMPERATURE = 1.0
TOP_P = 0.995
OUTPUT_PATH = "uncertain_resamples.jsonl"
EXCLUDED_LABELS = ["_", "SYM", "INTJ"]
MAX_PREFIXES = None  # optional cap for debugging
MAX_DATASET_ITEMS = None  # optional cap for scanning phase
RNG_SEED = 1234


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def prepare_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_uncertain_tokens(model, tokenizer, clf, excluded_ids, device) -> List[dict]:
    dataset = load_dataset("universal_dependencies", "en_ewt", split=SPLIT)
    candidates = []
    class_set = set(int(c) for c in clf.classes_)

    model.eval()
    total = MAX_DATASET_ITEMS if MAX_DATASET_ITEMS is not None else len(dataset)
    for idx, example in tqdm(
        enumerate(dataset),
        total=total,
        desc="Scanning validation",
    ):
        if MAX_DATASET_ITEMS is not None and idx >= MAX_DATASET_ITEMS:
            break
        text = example["text"]
        ud_tokens = example["tokens"]
        ud_labels = example["upos"]
        ud_heads = example["head"]

        tokenized, token_labels = tokenize_with_labels(
            text,
            ud_tokens,
            ud_labels,
            ud_heads,
            tokenizer,
        )
        input_ids = torch.tensor([tokenized["input_ids"]], device=device)
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[TARGET_LAYER].squeeze(0).detach().cpu()

        for pos, (token_id, label) in enumerate(zip(tokenized["input_ids"], token_labels)):
            if label is None or label in excluded_ids:
                continue
            if label not in class_set:
                continue
            feature = hidden_states[pos].float().unsqueeze(0).numpy()
            probs = clf.predict_proba(feature)[0]
            max_prob = float(probs.max())
            if max_prob >= CONF_THRESH:
                continue
            pred_idx = int(np.argmax(probs))
            pred_label = int(clf.classes_[pred_idx])

            prefix_ids = tokenized["input_ids"][: pos + 1]
            candidate = {
                "dataset_idx": idx,
                "sentence_text": text,
                "prefix_token_ids": list(prefix_ids),
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
                "target_position": pos,
                "target_token_id": token_id,
                "target_token_text": tokenizer.decode([token_id], skip_special_tokens=True),
                "true_label": int(label),
                "true_label_name": upos_to_tag_name(label),
                "probe_probs": probs.tolist(),
                "probe_pred_label": pred_label,
                "probe_pred_name": upos_to_tag_name(pred_label),
                "probe_max_prob": max_prob,
            }
            candidates.append(candidate)

            if MAX_PREFIXES is not None and len(candidates) >= MAX_PREFIXES:
                return candidates
    return candidates


def sample_completions(model, tokenizer, prefix_ids: List[int], rng_seed_offset: int, device):
    prefix_tensor = torch.tensor(prefix_ids, device=device).unsqueeze(0)
    prefix_len = prefix_tensor.shape[1]

    completions = []
    remaining = NUM_SAMPLES

    while remaining > 0:
        chunk = min(CHUNK_SIZE, remaining)
        input_batch = prefix_tensor.repeat(chunk, 1)
        attention_mask = torch.ones_like(input_batch)
        chunk_seed = RNG_SEED + rng_seed_offset * 1000 + (NUM_SAMPLES - remaining)
        torch.manual_seed(chunk_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(chunk_seed)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_batch,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = outputs[:, prefix_len:]
        for seq in new_tokens:
            comp_ids = seq.tolist()
            comp_text = tokenizer.decode(seq, skip_special_tokens=True)
            completions.append({"token_ids": comp_ids, "text": comp_text})
        remaining -= chunk
    return completions


def main():
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    device = prepare_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ensure_pad_token(tokenizer)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    clf = joblib.load(PROBE_PATH)
    excluded_label_ids = [tag_name_to_upos(label) for label in EXCLUDED_LABELS]

    candidates = collect_uncertain_tokens(model, tokenizer, clf, excluded_label_ids, device)
    if not candidates:
        print("No tokens fell below the confidence threshold; nothing to sample.")
        return
    print(f"Collected {len(candidates)} uncertain prefixes (threshold={CONF_THRESH:.2f}).")

    metadata = {
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "split": SPLIT,
        "probe_path": PROBE_PATH,
        "confidence_threshold": CONF_THRESH,
        "num_samples_per_prefix": NUM_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "excluded_labels": EXCLUDED_LABELS,
        "probe_classes": [int(c) for c in clf.classes_],
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"metadata": metadata}, ensure_ascii=False) + "\n")
        for idx, candidate in enumerate(tqdm(candidates, desc="Sampling completions")):
            completions = sample_completions(model, tokenizer, candidate["prefix_token_ids"], idx, device)
            record = {
                "candidate": candidate,
                "completions": completions,
                "generation_params": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "num_samples": NUM_SAMPLES,
                    "seed": RNG_SEED + idx,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
