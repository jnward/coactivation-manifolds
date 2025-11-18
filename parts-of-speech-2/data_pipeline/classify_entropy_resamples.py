from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import jsonlines as jsonl
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

# Config
SPACY_MODEL = "en_core_web_trf"
TOKENIZER_NAME = "google/gemma-2-2b"
GLOBAL_BATCH_SIZE = 2048  # completions per spaCy chunk
PIPE_BATCH_SIZE = 64  # spaCy internal batch size
RECORD_BUFFER = 256  # candidate records per buffer before writing
MAX_RECORDS = None  # optional cap per shard


spacy.prefer_gpu()


@dataclass
class CompletionJob:
    candidate_idx: int
    completion_idx: int
    text: str
    start: int
    end: int


def load_spacy_model(model_name: str):
    spacy.require_gpu()
    return spacy.load(model_name)


def compute_target_offsets(tokenizer, candidate) -> Tuple[str, int, int]:
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


def build_jobs(records: List[dict], tokenizer) -> List[CompletionJob]:
    jobs: List[CompletionJob] = []
    for cand_idx, rec in enumerate(records):
        candidate = rec["candidate"]
        completions = rec.get("completions", [])
        if not completions:
            continue
        prefix_text, start, end = compute_target_offsets(tokenizer, candidate)
        for comp_idx, completion in enumerate(completions):
            text = prefix_text + completion.get("text", "")
            jobs.append(
                CompletionJob(
                    candidate_idx=cand_idx,
                    completion_idx=comp_idx,
                    text=text,
                    start=start,
                    end=end,
                )
            )
    return jobs


def process_jobs(jobs: List[CompletionJob], nlp, records: List[dict]):
    total = len(jobs)
    if total == 0:
        return

    for offset in range(0, total, GLOBAL_BATCH_SIZE):
        chunk = jobs[offset : offset + GLOBAL_BATCH_SIZE]
        texts = [job.text for job in chunk]
        docs = list(nlp.pipe(texts, batch_size=PIPE_BATCH_SIZE))
        for job, doc in zip(chunk, docs):
            completion = records[job.candidate_idx]["completions"][job.completion_idx]
            if job.start < 0 or job.end > len(job.text) or job.start >= job.end:
                # Invalid span; skip with None labels
                completion["spacy_pos"] = None
                completion["spacy_tag"] = None
                continue
            span = doc.char_span(job.start, job.end, alignment_mode="expand")
            if span is None or not span:
                completion["spacy_pos"] = None
                completion["spacy_tag"] = None
                continue
            token = span[0]
            completion["spacy_pos"] = token.pos_
            completion["spacy_tag"] = token.tag_


def pre_count_records(path: Path) -> int:
    count = 0
    with jsonl.open(path) as reader:
        for obj in reader:
            if isinstance(obj, dict) and "candidate" in obj:
                count += 1
            if MAX_RECORDS is not None and count >= MAX_RECORDS:
                break
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate entropy resamples with spaCy POS tags.")
    parser.add_argument("--input", type=Path, required=True, help="Input shard JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with spaCy tags")
    return parser.parse_args()


def main():
    args = parse_args()
    nlp = load_spacy_model(SPACY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    total_records = pre_count_records(args.input)

    with jsonl.open(args.input) as reader, jsonl.open(args.output, "w") as writer:
        metadata = None
        buffer: List[dict] = []
        pbar = tqdm(total=total_records, desc="Tagging candidates")

        for idx, obj in enumerate(reader):
            if isinstance(obj, dict) and "metadata" in obj:
                metadata = obj
                writer.write(metadata)
                continue
            if MAX_RECORDS is not None and len(buffer) >= MAX_RECORDS:
                break
            if not isinstance(obj, dict) or "candidate" not in obj:
                continue
            buffer.append(obj)
            if len(buffer) >= RECORD_BUFFER:
                jobs = build_jobs(buffer, tokenizer)
                process_jobs(jobs, nlp, buffer)
                for rec in buffer:
                    writer.write(rec)
                    pbar.update(1)
                buffer.clear()

        # Flush remaining
        if buffer:
            jobs = build_jobs(buffer, tokenizer)
            process_jobs(jobs, nlp, buffer)
            for rec in buffer:
                writer.write(rec)
                pbar.update(1)
            buffer.clear()

    print(f"Wrote annotated records to {args.output}")


if __name__ == "__main__":
    main()
