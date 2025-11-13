from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import jsonlines as jsonl
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_PATH = "test_resamples.jsonl"
OUTPUT_PATH = "output_with_spacy_pos.jsonl"
SPACY_MODEL = "en_core_web_trf"
TOKENIZER_NAME = "google/gemma-2-2b"
BATCH_SIZE = 16

# Prefer GPU if available (spaCy silently falls back to CPU).
spacy.prefer_gpu()


def load_spacy_model(model_name: str):
    """Load spaCy model, downloading it if missing."""
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        print(f"spaCy model '{model_name}' not found. Downloading...")
        download(model_name)
        return spacy.load(model_name)


def read_records(path: Path) -> Tuple[Optional[dict], List[dict]]:
    """Read metadata (if present) and candidate records from JSONL."""
    metadata = None
    records: List[dict] = []
    with jsonl.open(path) as reader:
        for obj in reader:
            if isinstance(obj, dict) and "metadata" in obj:
                metadata = obj
                continue
            if isinstance(obj, dict) and "candidate" in obj:
                records.append(obj)
    return metadata, records


def compute_target_offsets(tokenizer, candidate) -> Tuple[str, int, int]:
    """Return prefix text and char offsets for the target token."""
    decode_kwargs = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    prefix_with_target = candidate["prefix_text"]
    recomputed_prefix = tokenizer.decode(candidate["prefix_token_ids"], **decode_kwargs)
    if prefix_with_target != recomputed_prefix:
        prefix_with_target = recomputed_prefix
        candidate["prefix_text"] = prefix_with_target
    prefix_without_target = tokenizer.decode(candidate["prefix_token_ids"][:-1], **decode_kwargs)
    start = len(prefix_without_target)
    end = len(prefix_with_target)
    return prefix_with_target, start, end


def annotate_records(records: List[dict], nlp, tokenizer):
    for rec in tqdm(records, desc="Annotating completions"):
        candidate = rec["candidate"]
        completions = rec.get("completions", [])
        if not completions:
            continue

        prefix_text, start, end = compute_target_offsets(tokenizer, candidate)
        completion_texts = [prefix_text + completion.get("text", "") for completion in completions]
        docs = list(nlp.pipe(completion_texts, batch_size=BATCH_SIZE))

        for doc, completion in zip(docs, completions):
            span = doc.char_span(start, end, alignment_mode="expand")
            if span is None or not span:
                completion["spacy_pos"] = None
                completion["spacy_tag"] = None
                continue
            token = span[0]
            completion["spacy_pos"] = token.pos_
            completion["spacy_tag"] = token.tag_


def write_records(path: Path, metadata: Optional[dict], records: Iterable[dict]):
    with jsonl.open(path, "w") as writer:
        if metadata:
            writer.write(metadata)
        for rec in records:
            writer.write(rec)


def main():
    nlp = load_spacy_model(SPACY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    metadata, records = read_records(Path(INPUT_PATH))
    if not records:
        print("No candidate records found in input.")
        return

    annotate_records(records, nlp, tokenizer)
    write_records(Path(OUTPUT_PATH), metadata, records)
    print(f"Wrote {len(records)} annotated candidate records (per-completion tags) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

