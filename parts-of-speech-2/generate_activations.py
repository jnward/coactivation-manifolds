from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

from utils import tokenize_with_labels, upos_to_tag_name

MODEL_NAME = "google/gemma-2-2b"
LAYER = 8
SPLIT = "train"
SHUFFLE = True
SEED = 42
MAX_SAMPLES = float("inf")
# MAX_SAMPLES = 256
OUTPUT_PATH = f"activations_{SPLIT}_{LAYER}.npz"

def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()

    label_counts = Counter()
    features = []
    labels = []
    positions = []

    dataset = load_dataset(
        "universal_dependencies",
        "en_ewt",
        split=SPLIT,
    )
    max_samples = min(len(dataset), MAX_SAMPLES)

    for i, example in enumerate(tqdm(dataset.shuffle(seed=SEED) if SHUFFLE else dataset)):
        if i >= max_samples:
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
        label_counts.update(label for label in token_labels)

        input_ids = torch.tensor([tokenized["input_ids"]]).cuda()
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[LAYER].squeeze(0).detach().cpu().float()

        for position_idx, (token_id, label, hidden_state) in enumerate(zip(tokenized["input_ids"], token_labels, hidden_states)):
            token_str = tokenizer.decode([token_id])
            tag_name = upos_to_tag_name(label)
            # print(f"Token: {token_str}\tTag: {tag_name}")
            if label is None:
                continue
            features.append(hidden_state.numpy())
            labels.append(label)
            positions.append(position_idx)

    print("Label counts:")
    for label, count in label_counts.items():
        tag_name = upos_to_tag_name(label)
        print(f"{tag_name}: {count}")

    if not features:
        raise RuntimeError("No labeled tokens were collected; cannot save activations.")
    
    X = np.stack(features).astype(np.float32, copy=False)
    y = np.array(labels, dtype=np.int64)
    pos = np.array(positions, dtype=np.int32)

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=y,
        positions=pos,
        layer=LAYER,
        model=MODEL_NAME,
    )

if __name__ == "__main__":
    main()
