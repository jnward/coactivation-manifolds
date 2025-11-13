from __future__ import annotations

from pathlib import Path
from typing import Dict

import jsonlines as jsonl
import numpy as np

from utils import TAG_NAMES

INPUT_PATH = "output_with_spacy_pos.jsonl"
OUTPUT_PATH = "probe_vs_spacy.jsonl"
EPS = 1e-8


def vector_from_probe(candidate: dict, classes: list[int]) -> np.ndarray:
    vec = np.zeros(len(TAG_NAMES), dtype=np.float64)
    probs = candidate.get("probe_probs", [])
    for prob, label in zip(probs, classes):
        if label < len(vec):
            vec[label] = prob
    if vec.sum() == 0:
        raise ValueError("Probe probabilities missing or summed to zero.")
    return vec


def vector_from_spacy(completions: list[dict]) -> tuple[np.ndarray, int]:
    vec = np.zeros(len(TAG_NAMES), dtype=np.float64)
    count = 0
    for comp in completions:
        label = comp.get("spacy_pos")
        if not label:
            continue
        try:
            idx = TAG_NAMES.index(label)
        except ValueError:
            continue
        vec[idx] += 1.0
        count += 1
    if count > 0:
        vec /= count
    return vec, count


def smooth_distribution(dist: np.ndarray) -> np.ndarray:
    smoothed = dist.copy()
    smoothed[smoothed == 0] = EPS
    smoothed /= smoothed.sum()
    return smoothed


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p_safe = smooth_distribution(p)
    q_safe = smooth_distribution(q)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def dist_to_dict(dist: np.ndarray) -> Dict[str, float]:
    return {tag: float(value) for tag, value in zip(TAG_NAMES, dist)}


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    num_written = 0
    num_skipped = 0

    with jsonl.open(input_path) as reader, jsonl.open(output_path, "w") as writer:
        metadata = None
        probe_classes = None

        for obj in reader:
            if metadata is None and isinstance(obj, dict) and "metadata" in obj:
                metadata = obj["metadata"]
                probe_classes = metadata.get("probe_classes")
                output_meta = {
                    "source": INPUT_PATH,
                    "epsilon": EPS,
                    "probe_classes": probe_classes,
                }
                writer.write({"metadata": output_meta})
                continue

            if probe_classes is None:
                raise ValueError("Missing metadata with probe_classes before data records.")

            candidate = obj.get("candidate")
            completions = obj.get("completions", [])
            if not candidate or not completions:
                num_skipped += 1
                continue

            spacy_vec, spacy_count = vector_from_spacy(completions)
            if spacy_count == 0:
                num_skipped += 1
                continue

            probe_vec = vector_from_probe(candidate, probe_classes)
            kl = kl_divergence(probe_vec, spacy_vec)

            writer.write(
                {
                    "dataset_idx": candidate.get("dataset_idx"),
                    "target_token": candidate.get("target_token_text"),
                    "probe_pred": candidate.get("probe_pred_name"),
                    "probe_distribution": dist_to_dict(probe_vec),
                    "spacy_distribution": dist_to_dict(spacy_vec),
                    "num_spacy_samples": spacy_count,
                    "kl_divergence": kl,
                }
            )
            num_written += 1

    print(f"Wrote {num_written} summaries to {OUTPUT_PATH} (skipped {num_skipped}).")


if __name__ == "__main__":
    main()
