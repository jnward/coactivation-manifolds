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

PROBE_PATH = Path("spacy_retagged/spacy_probe.joblib")
MODEL_NAME = "google/gemma-2-2b"
SPLIT = "train"
TARGET_LAYER = 12
CONF_THRESH = 0.95
NUM_SAMPLES = 1000
CHUNK_SIZE = 100
MAX_NEW_TOKENS = 12
TEMPERATURE = 1.0
TOP_P = 0.995
OUTPUT_PATH = Path("uncertain_train_spacy_resamples.jsonl")
MAX_PREFIXES = None
MAX_DATASET_ITEMS = None
RNG_SEED = 1234
FILTER_BY_PROBE = False


def ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def prepare_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("GPU (CUDA or MPS) is required for resampling.")


def load_retagged(split: str):
    path = Path("spacy_retagged") / f"spacy_tags_{split}.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as f:
        header = json.loads(f.readline())
        metadata = header["metadata"]
        tag_names = metadata["tag_names"]
        for line in f:
            records.append(json.loads(line))
    return metadata, tag_names, records


def find_token_label(offset, char_tags, unknown_id):
    start, end = offset
    if start == end or end > len(char_tags):
        return None
    for idx in range(start, end):
        tag_id = char_tags[idx]
        if tag_id != unknown_id:
            return tag_id
    return None


def collect_uncertain_tokens(model, tokenizer, clf, tag_names, records, device) -> List[dict]:
    candidates = []
    class_set = set(int(c) for c in clf.classes_)
    unknown_id = tag_names.index("_") if "_" in tag_names else None

    model.eval()
    total = min(len(records), MAX_DATASET_ITEMS) if MAX_DATASET_ITEMS is not None else len(records)

    for idx, record in enumerate(tqdm(records[:total], desc="Scanning retagged data", total=total)):
        text = record["text"]
        char_tags = record["char_tags"]
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"].to(device)
        offsets = encoded["offset_mapping"][0].tolist()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[TARGET_LAYER].squeeze(0).detach().cpu()

        for pos, offset in enumerate(offsets):
            label = find_token_label(offset, char_tags, unknown_id)
            if label is None or label in clf.classes_ and label not in class_set:
                continue
            if label not in class_set:
                continue
            feature = hidden_states[pos].float().unsqueeze(0).numpy()
            probs = clf.predict_proba(feature)[0]
            max_prob = float(probs.max())
            if FILTER_BY_PROBE and max_prob >= CONF_THRESH:
                continue
            pred_idx = int(np.argmax(probs))
            pred_label = int(clf.classes_[pred_idx])

            prefix_ids = encoded["input_ids"][0][: pos + 1].tolist()
            candidate = {
                "dataset_idx": record["dataset_idx"],
                "sentence_text": text,
                "prefix_token_ids": prefix_ids,
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
                "target_position": pos,
                "target_token_id": int(encoded["input_ids"][0][pos]),
                "target_token_text": tokenizer.decode([encoded["input_ids"][0][pos]], skip_special_tokens=True),
                "true_label": int(label),
                "true_label_name": tag_names[label] if 0 <= label < len(tag_names) else None,
                "probe_probs": probs.tolist(),
                "probe_pred_label": pred_label,
                "probe_pred_name": tag_names[pred_label],
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
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()

    probe_bundle = joblib.load(PROBE_PATH)
    clf = probe_bundle["model"]
    tag_names = probe_bundle["tag_names"]

    _, retag_tag_names, records = load_retagged(SPLIT)
    if tag_names != retag_tag_names:
        raise ValueError("Probe tag_names do not match retag metadata.")

    candidates = collect_uncertain_tokens(model, tokenizer, clf, tag_names, records, device)
    if not candidates:
        if FILTER_BY_PROBE:
            print("No tokens fell below the confidence threshold; nothing to sample.")
        else:
            print("No tokens available to resample.")
        return
    if FILTER_BY_PROBE:
        print(f"Collected {len(candidates)} uncertain prefixes (threshold={CONF_THRESH:.2f}).")
    else:
        print(f"Collected {len(candidates)} prefixes (no probe filtering).")

    metadata = {
        "model_name": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "split": SPLIT,
        "probe_path": str(PROBE_PATH),
        "confidence_threshold": CONF_THRESH,
        "filter_by_probe": FILTER_BY_PROBE,
        "num_samples_per_prefix": NUM_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "probe_classes": [int(c) for c in clf.classes_],
        "tag_names": tag_names,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
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
