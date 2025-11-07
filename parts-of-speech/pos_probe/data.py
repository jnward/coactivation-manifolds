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

    keep_columns = {"tokens", "upos", "text"}
    remove_columns = [
        col for col in dataset["train"].column_names if col not in keep_columns
    ]

    def _batch_tokenize(batch: Dict[str, List[Sequence[str]]]) -> Dict[str, List]:
        encodings = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

        batch_input_ids = encodings["input_ids"]
        batch_attention = encodings["attention_mask"]

        batch_labels: List[List[int]] = []
        batch_label_mask: List[List[int]] = []

        for i in range(len(batch_input_ids)):
            word_ids = encodings.word_ids(i)
            upos_tags = _normalize_tag_sequence(batch["upos"][i], tag_names)

            seq_len = len(batch_input_ids[i])
            labels = [-100] * seq_len
            mask = [0] * seq_len

            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                is_last = token_idx == seq_len - 1 or word_ids[token_idx + 1] != word_idx
                if not is_last:
                    continue
                tag = map_tag(upos_tags[word_idx])
                labels[token_idx] = TAG_TO_ID[tag]
                mask[token_idx] = 1

            batch_labels.append(labels)
            batch_label_mask.append(mask)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": batch_labels,
            "label_mask": batch_label_mask,
        }

    tokenized = dataset.map(
        _batch_tokenize,
        batched=True,
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
        batch["tokens"] = [f["tokens"] for f in features]
        batch["upos"] = [f["upos"] for f in features]
        batch["text"] = [f.get("text", "") for f in features]
        return batch
