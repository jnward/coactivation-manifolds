"""Shared constants for the POS probing project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

TAG_TO_ID = {tag: idx for idx, tag in enumerate(POS_TAGS)}
ID_TO_TAG = {idx: tag for tag, idx in TAG_TO_ID.items()}

DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_DATASET = ("universal_dependencies", "en_ewt")
DEFAULT_LAYER_INDEX = 12
DEFAULT_ACTIVATION_CACHE_DIR = Path("parts-of-speech/cache")
DEFAULT_OUTPUT_DIR = Path("parts-of-speech/runs")


@dataclass
class ProbeConfig:
    """Summary of the configuration used to train/evaluate a probe."""

    model_name: str = DEFAULT_MODEL_NAME
    layer_index: int = DEFAULT_LAYER_INDEX
    classification_mode: str = "multiclass"
    normalization: str = "softmax"
    tag_set: tuple[str, ...] = tuple(POS_TAGS)
    disabled_tags: tuple[str, ...] = ()
    max_length: int = 256
    activation_cache_dir: Path = DEFAULT_ACTIVATION_CACHE_DIR

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["tag_set"] = list(self.tag_set)
        data["disabled_tags"] = list(self.disabled_tags)
        data["activation_cache_dir"] = str(self.activation_cache_dir)
        return data
