"""Score definition choices for collected feature snippets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign definition probabilities to snippets")
    parser.add_argument("snippets_path", type=Path, help="Parquet created by collect_joint_snippets.py")
    parser.add_argument(
        "--choice",
        action="append",
        required=True,
        help="Definition option in the form LABEL=Definition text (repeat per choice)",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-2-2b-it",
        help="Instruction-tuned model for scoring",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model inference (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Prompts per forward pass")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: <input>_scored.parquet)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    return parser.parse_args()


def parse_choices(raw: List[str]) -> List[Tuple[str, str]]:
    parsed: List[Tuple[str, str]] = []
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"Choice must be LABEL=definition, got: {entry}")
        label, definition = entry.split("=", 1)
        label = label.strip()
        definition = definition.strip()
        if not label:
            raise ValueError("Choice label cannot be empty")
        parsed.append((label, definition))
    return parsed


def build_prompt(snippet: str, choices: List[Tuple[str, str]]) -> str:
    lines = [
        "Here is a text snippet where the target word appears:",
        snippet,
        "",
        "Which definition best fits its usage in the snippet?",
    ]
    for label, definition in choices:
        lines.append(f"{label}: {definition}")
    lines.append("Respond with only the single-letter label (A, B, or C).")
    return "\n".join(lines)


def encode_batch(tokenizer, prompts: List[str]):
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    encodings = [
        tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        for conv in conversations
    ]
    encodings = [{"input_ids": enc.squeeze(0)} for enc in encodings]
    batch = tokenizer.pad(encodings, padding=True, return_tensors="pt")
    return batch


def choice_token_ids(tokenizer, choices: List[Tuple[str, str]]) -> List[int]:
    ids: List[int] = []
    for label, _ in choices:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Choice label '{label}' does not tokenize to a single token: {token_ids}")
        ids.append(token_ids[0])
    return ids


def score_batches(
    model,
    tokenizer,
    prompts: List[str],
    choices: List[Tuple[str, str]],
    batch_size: int,
    *,
    show_progress: bool,
) -> Tuple[List[List[float]], List[List[float]]]:
    label_ids = choice_token_ids(tokenizer, choices)
    probabilities: List[List[float]] = []
    logits_store: List[List[float]] = []
    device = next(model.parameters()).device

    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(prompts) + batch_size - 1) // batch_size, desc="Scoring", unit="batch")
    for start in iterator:
        batch_prompts = prompts[start : start + batch_size]
        batch_inputs = encode_batch(tokenizer, batch_prompts)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        with torch.no_grad():
            outputs = model(**batch_inputs)
            next_logits = outputs.logits[:, -1, :]
            label_logits = torch.stack([next_logits[:, idx] for idx in label_ids], dim=-1)
            probs = torch.softmax(label_logits, dim=-1)
        probabilities.extend(probs.cpu().tolist())
        logits_store.extend(label_logits.cpu().tolist())

    return probabilities, logits_store


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_scored.parquet")


def main() -> None:
    args = parse_args()
    choices = parse_choices(args.choice)
    if len(choices) < 2:
        raise ValueError("At least two choices are required")

    snippets_table = pq.read_table(args.snippets_path)
    snippets = snippets_table.column("token_text").to_pylist()
    prompts = [build_prompt(snippet, choices) for snippet in snippets]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    model.to(args.device)
    model.eval()

    probabilities, logits_store = score_batches(
        model,
        tokenizer,
        prompts,
        choices,
        args.batch_size,
        show_progress=not args.no_progress,
    )

    prob_array = pa.array(probabilities, type=pa.list_(pa.float32()))
    logit_array = pa.array(logits_store, type=pa.list_(pa.float32()))
    table = snippets_table.append_column("definition_probabilities", prob_array)
    table = table.append_column("definition_logits", logit_array)

    metadata = table.schema.metadata or {}
    metadata = dict(metadata)
    metadata.update(
        {
            b"choice_labels": json.dumps([label for label, _ in choices]).encode(),
            b"choice_definitions": json.dumps({label: definition for label, definition in choices}).encode(),
            b"model_name": args.model_name.encode(),
        }
    )
    table = table.replace_schema_metadata(metadata)

    output_path = args.output or default_output_path(args.snippets_path)
    pq.write_table(table, output_path, compression="zstd")
    print(f"Wrote scored snippets to {output_path}")


if __name__ == "__main__":
    main()
