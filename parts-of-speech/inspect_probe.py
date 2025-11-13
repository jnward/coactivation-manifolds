#!/usr/bin/env python
"""Inspect probe predictions on sample UD sentences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from pos_probe import activations, constants, data, probe, utils  # noqa: E402
from pos_probe.utils import normalize_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect POS probe predictions.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to probe.pt")
    parser.add_argument("--model-name", default=constants.DEFAULT_MODEL_NAME)
    parser.add_argument("--layer-index", type=int, default=constants.DEFAULT_LAYER_INDEX)
    parser.add_argument("--split", default="test", help="UD split to sample from.")
    parser.add_argument("--num-sentences", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-proc", type=int, default=2)
    parser.add_argument("--normalization", choices=["softmax", "relu", "both"], default="both")
    parser.add_argument(
        "--drop-tags",
        type=str,
        default="",
        help="Comma-separated POS tags to mask at inference (e.g., INTJ).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = data.load_tokenizer(args.model_name)
    raw_dataset, tag_names = data.load_ud_dataset()
    tag_lookup = {idx: name for idx, name in enumerate(tag_names)}
    dataset = data.tokenize_and_align(
        raw_dataset,
        tokenizer,
        tag_names=tag_names,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )

    loader = DataLoader(
        dataset[args.split],
        batch_size=1,
        shuffle=False,
        collate_fn=data.POSDataCollator(tokenizer),
    )

    probe_model, metadata = probe.load_probe(args.weights, map_location="cpu")
    probe_model.to(args.device)
    probe_model.eval()

    model = activations.load_model(args.model_name, device=args.device)

    normalizations = ["softmax", "relu"] if args.normalization == "both" else [args.normalization]
    active_tags = metadata.get("active_tags") or list(constants.POS_TAGS[: probe_model.linear.out_features])
    active_tags = list(active_tags)[: probe_model.linear.out_features]
    tag_to_idx = {tag: idx for idx, tag in enumerate(active_tags)}

    drop_tags = {
        tag.strip().upper()
        for tag in args.drop_tags.split(",")
        if tag.strip()
    }
    drop_indices = sorted(tag_to_idx[tag] for tag in drop_tags if tag in tag_to_idx)
    if drop_tags:
        missing = drop_tags.difference(tag_to_idx)
        if missing:
            print(f"Warning: dropped tags not in probe head: {', '.join(sorted(missing))}")

    shown = 0
    for batch in loader:
        if shown >= args.num_sentences:
            break
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"].to(args.device),
                attention_mask=batch["attention_mask"].to(args.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = outputs.hidden_states[args.layer_index + 1][0]
        mask = batch["label_mask"][0].bool()
        subtoken_texts = tokenizer.convert_ids_to_tokens(
            batch["input_ids"][0].tolist(), skip_special_tokens=False
        )
        text = batch["text"][0]
        words_raw = batch["tokens"][0]
        upos_raw = batch["upos"][0]
        heads = batch["heads"][0]
        normalized_tags = [
            tag_lookup.get(int(tag), str(tag)) if isinstance(tag, (int, float)) else tag
            for tag in upos_raw
        ]
        filtered_words, filtered_tags = data.filter_tokens_by_head(
            words_raw,
            normalized_tags,
            heads,
        )
        char_map = data.build_char_map(text, filtered_words)
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        offsets = encoding["offset_mapping"]
        subtoken_word_idx = []
        for start, end in offsets:
            if start == end or not char_map:
                subtoken_word_idx.append(None)
                continue
            char_idx = min(end - 1, len(char_map) - 1)
            word_index = char_map[char_idx]
            subtoken_word_idx.append(word_index if word_index >= 0 else None)

        vecs = hidden[mask].to(args.device, dtype=torch.float32)
        logits = probe_model(vecs)

        print(f"\nSentence {shown + 1}: {text}")

        mask_positions = mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        for vec_idx, sub_pos in enumerate(mask_positions):
            word_index = subtoken_word_idx[sub_pos] if sub_pos < len(subtoken_word_idx) else None
            if word_index is None or word_index >= len(filtered_words):
                continue
            word = subtoken_texts[sub_pos]
            gold_tag = filtered_tags[word_index].strip().upper()
            if gold_tag not in tag_to_idx:
                print(f"{word:<15} gold={gold_tag:<6} | tag not in probe head (skipped)")
                continue
            probs_by_mode = {}
            for mode in normalizations:
                probs = normalize_logits(logits[vec_idx][None, :], mode=mode)[0]
                if drop_indices:
                    probs_adj = probs.clone()
                    probs_adj[drop_indices] = 0.0
                    if mode == "softmax":
                        probs_adj = probs_adj / probs_adj.sum() if probs_adj.sum() > 0 else probs_adj
                    else:
                        total = probs_adj.sum()
                        probs_adj = probs_adj / total if total > 0 else probs_adj
                    probs = probs_adj
                probs_by_mode[mode] = probs

            if drop_tags and gold_tag in drop_tags:
                print(f"{word:<15} gold={gold_tag:<6} | dropped tag (skipped)")
                continue

            print(f"{word:<15} gold={gold_tag:<6}", end="")
            for mode, probs in probs_by_mode.items():
                pred_idx = probs.argmax().item()
                pred_tag = active_tags[pred_idx]
                pred_prob = probs[pred_idx].item()
                marker = "✓" if pred_tag == gold_tag else "✗"
                print(f" | {mode}: {pred_tag} ({pred_prob:.2f}) {marker}", end="")
            print()

        shown += 1


if __name__ == "__main__":
    main()
