"""Dataset loading and token-to-subtoken alignment utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import datasets
from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import torch

from .constants import POS_TAGS, TAG_TO_ID, DEFAULT_DATASET

ALT_TAG_MAP = {
    "_": "X",  # UD sometimes uses "_" placeholders; collapse into X (other).
}


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the Gemma tokenizer with sensible defaults for alignment."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "right"
    tokenizer.model_max_length = min(tokenizer.model_max_length, 2048)
    return tokenizer


def load_ud_dataset() -> tuple[DatasetDict, List[str]]:
    """Fetch the Universal Dependencies English EWT dataset and tag names."""

    name, config = DEFAULT_DATASET
    dataset = datasets.load_dataset(name, config)
    feature = dataset["train"].features["upos"].feature  # Sequence(ClassLabel)
    tag_names = list(feature.names)
    return dataset, tag_names


def map_tag(tag: str) -> str:
    mapped = ALT_TAG_MAP.get(tag, tag)
    if mapped not in TAG_TO_ID:
        raise ValueError(f"Unexpected POS tag '{tag}'")
    return mapped


def _normalize_tag_sequence(tags: Sequence, tag_names: Sequence[str]) -> List[str]:
    """Convert ClassLabel indices to strings if needed."""

    if not tags:
        return list(tags)
    first = tags[0]
    if isinstance(first, str):
        return list(tags)
    return [tag_names[int(idx)] for idx in tags]


def tokenize_and_align(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    tag_names: Sequence[str],
    *,
    max_length: int = 256,
    num_proc: int | None = None,
) -> DatasetDict:
    """Tokenize sentences and align POS tags to final subtokens."""

    remove_columns = dataset["train"].column_names

    tokenized = dataset.map(
        lambda ex: align_example(
            ex, tokenizer, tag_names, max_length
        ),
        batched=False,
        remove_columns=remove_columns,
        num_proc=num_proc,
        desc="Tokenizing + aligning UD POS tags",
    )
    return tokenized


class POSDataCollator:
    """Pads tokenized sequences and keeps label masks aligned."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, label_pad: int = -100):
        self.tokenizer = tokenizer
        self.label_pad = label_pad

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_inputs = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]
        batch = self.tokenizer.pad(batch_inputs, padding=True, return_tensors="pt")
        batch_size, seq_len = batch["input_ids"].shape

        labels = torch.full((batch_size, seq_len), self.label_pad, dtype=torch.long)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        for i, feature in enumerate(features):
            length = len(feature["input_ids"])
            labels[i, :length] = torch.tensor(feature["labels"], dtype=torch.long)
            mask[i, :length] = torch.tensor(feature["label_mask"], dtype=torch.bool)

        batch["labels"] = labels
        batch["label_mask"] = mask

        extra_keys = set(features[0].keys()) - {
            "input_ids",
            "attention_mask",
            "labels",
            "label_mask",
        }
        for key in extra_keys:
            batch[key] = [f[key] for f in features]
        return batch


def filter_tokens_by_head(
    tokens: Sequence[str],
    upos_tags: Sequence[str],
    heads: Sequence,
) -> tuple[list[str], list[str]]:
    filtered_tokens: list[str] = []
    filtered_tags: list[str] = []
    for token, tag, head in zip(tokens, upos_tags, heads):
        if head == "None":
            continue
        filtered_tokens.append(token)
        filtered_tags.append(tag)
    return filtered_tokens, filtered_tags


def build_char_map(text: str, tokens: Sequence[str]) -> list[int]:
    char_to_word = [-1] * len(text)
    text_pos = 0
    word_idx = 0
    char_in_word = 0
    while text_pos < len(text) and word_idx < len(tokens):
        current = tokens[word_idx]
        if char_in_word >= len(current):
            word_idx += 1
            char_in_word = 0
            continue
        if text[text_pos] == current[char_in_word]:
            char_to_word[text_pos] = word_idx
            char_in_word += 1
            text_pos += 1
        else:
            text_pos += 1
    return char_to_word


def align_example(
    example: Dict[str, Sequence],
    tokenizer: PreTrainedTokenizerBase,
    tag_names: Sequence[str],
    max_length: int,
) -> Dict[str, Sequence]:
    text: str = example["text"]
    tokens_raw: Sequence[str] = example["tokens"]
    upos_raw = _normalize_tag_sequence(example["upos"], tag_names)
    heads = example.get("head") or example.get("heads") or ["0"] * len(tokens_raw)

    tokens, upos_tags = filter_tokens_by_head(tokens_raw, upos_raw, heads)

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids = encoding["input_ids"]
    attention = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    labels = [-100] * len(input_ids)
    mask = [0] * len(input_ids)

    if tokens:
        char_map = build_char_map(text, tokens)
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            char_idx = min(end - 1, len(char_map) - 1) if len(char_map) > 0 else -1
            token_idx = char_map[char_idx] if char_idx >= 0 else -1
            if token_idx == -1:
                continue
            tag = map_tag(upos_tags[token_idx])
            labels[idx] = TAG_TO_ID[tag]
            mask[idx] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "label_mask": mask,
        "tokens": list(tokens_raw),
        "upos": list(upos_raw),
        "heads": list(heads),
        "text": text,
    }
