import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b"
LAYER = 12
SPLIT = "train"
RETAG_DIR = Path("spacy_retagged")
SHUFFLE = False
SEED = 42
MAX_SAMPLES = None  # number of sentences; None = all
OUTPUT_PATH = RETAG_DIR / f"spacy_activations_{SPLIT}_{LAYER}.npz"


def load_retagged(split: str):
    path = RETAG_DIR / f"spacy_tags_{split}.jsonl"
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


def main():
    metadata, tag_names, records = load_retagged(SPLIT)
    if SHUFFLE:
        random.Random(SEED).shuffle(records)
    if MAX_SAMPLES is not None:
        records = records[:MAX_SAMPLES]

    unknown_id = tag_names.index("_") if "_" in tag_names else None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for activation generation.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    features = []
    labels = []
    positions = []

    for record in tqdm(records, desc=f"Processing {SPLIT}"):
        text = record["text"]
        char_tags = record["char_tags"]
        if len(char_tags) != len(text):
            raise ValueError(f"Char tag length mismatch for dataset_idx={record.get('dataset_idx')}")
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"].cuda()
        offsets = encoded["offset_mapping"][0].tolist()
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[LAYER].squeeze(0).cpu().float()

        for pos, offset in enumerate(offsets):
            label = find_token_label(offset, char_tags, unknown_id)
            if label is None:
                continue
            features.append(hidden_states[pos].numpy())
            labels.append(label)
            positions.append(pos)

    if not features:
        raise RuntimeError("No labeled tokens were collected; cannot save activations.")

    X = np.stack(features).astype(np.float32, copy=False)
    y = np.array(labels, dtype=np.int64)
    pos = np.array(positions, dtype=np.int32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        positions=pos,
        layer=LAYER,
        model=MODEL_NAME,
        split=SPLIT,
        tag_names=np.array(tag_names, dtype=object),
        metadata=json.dumps(metadata),
    )


if __name__ == "__main__":
    main()
