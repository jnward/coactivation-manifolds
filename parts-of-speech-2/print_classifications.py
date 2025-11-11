import random
import numpy as np
import torch
import joblib
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import tokenize_with_labels, upos_to_tag_name

PROBE_PATH = "probe.joblib"
MODEL_NAME = "google/gemma-2-2b"
LAYER_OF_INTEREST = 12
SPLIT = "test"
NUM_SENTENCES = 24
SEED = 0


def format_top_k(probs, classes, top_k=3):
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        (upos_to_tag_name(classes[idx]), probs[idx])
        for idx in top_indices
    ]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    clf = joblib.load(PROBE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).cuda()
    model.eval()

    dataset = load_dataset("universal_dependencies", "en_ewt", split=SPLIT)
    samples = list(dataset.shuffle(seed=SEED).select(range(NUM_SENTENCES)))

    for idx, example in enumerate(samples, 1):
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
        input_ids = torch.tensor([tokenized["input_ids"]]).cuda()
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[LAYER_OF_INTEREST].squeeze(0).detach().cpu().float().numpy()

        print(f"\nSentence {idx}: {text}")

        labeled_indices = [
            i
            for i, label in enumerate(token_labels)
            if label is not None and label != 13 and label in clf.classes_
        ]
        if not labeled_indices:
            print("  (No labeled tokens)")
            continue

        X = hidden_states[labeled_indices]
        y_true = [token_labels[i] for i in labeled_indices]
        tokens = [tokenizer.decode([tokenized["input_ids"][i]]) for i in labeled_indices]

        probs = clf.predict_proba(X)
        classes = clf.classes_
        class_indices = probs.argmax(axis=1)
        preds = classes[class_indices]

        for token, true_label, pred_label, prob_vec in zip(tokens, y_true, preds, probs):
            true_name = upos_to_tag_name(true_label)
            pred_name = upos_to_tag_name(pred_label)
            correct = true_label == pred_label
            status = "✓" if correct else "✗"
            print(f"  {token}: true={true_name}, pred={pred_name} {status}")
            if not correct:
                top_guesses = format_top_k(prob_vec, classes)
                guesses_str = ", ".join(f"{name}:{p:.2f}" for name, p in top_guesses)
                print(f"    Top guesses: {guesses_str}")


if __name__ == "__main__":
    main()
