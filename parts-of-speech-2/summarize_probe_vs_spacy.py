from __future__ import annotations

from pathlib import Path
from typing import Dict
import html as html_lib

import jsonlines as jsonl
import numpy as np

INPUT_PATH = "spacy_output_with_spacy_pos.jsonl"
OUTPUT_PATH = "spacy_probe_vs_spacy.jsonl"
HTML_OUTPUT = "spacy_probe_vs_spacy.html"
HTML_TOP_K = 50
EPS = 1e-8


def vector_from_probe(candidate: dict, classes: list[int], tag_names: list[str]) -> np.ndarray:
    vec = np.zeros(len(tag_names), dtype=np.float64)
    probs = candidate.get("probe_probs", [])
    for prob, label in zip(probs, classes):
        if label < len(vec):
            vec[label] = prob
    if vec.sum() == 0:
        raise ValueError("Probe probabilities missing or summed to zero.")
    return vec


def vector_from_spacy(completions: list[dict], tag_names: list[str]) -> tuple[np.ndarray, int]:
    vec = np.zeros(len(tag_names), dtype=np.float64)
    count = 0
    for comp in completions:
        label = comp.get("spacy_pos")
        if not label:
            continue
        try:
            idx = tag_names.index(label)
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


def dist_to_dict(dist: np.ndarray, tag_names: list[str]) -> Dict[str, float]:
    return {tag: float(value) for tag, value in zip(tag_names, dist)}


def top_tags(dist: np.ndarray, tag_names: list[str], k: int = 3) -> str:
    indices = np.argsort(dist)[::-1][:k]
    return ", ".join(f"{tag_names[i]} ({dist[i]*100:.1f}%)" for i in indices)


def render_html(rows: list[dict]):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["kl_divergence"], reverse=True)
    rows = rows[: min(len(rows), HTML_TOP_K)]
    html_rows = []
    for r in rows:
        html_rows.append(
            "<tr>"
            f"<td>{r['dataset_idx']}</td>"
            f"<td>{html_lib.escape(r['target_token'])}</td>"
            f"<td>{html_lib.escape(r['probe_pred'])}</td>"
            f"<td>{r['kl_divergence']:.4f}</td>"
            f"<td>{html_lib.escape(r['probe_top'])}</td>"
            f"<td>{html_lib.escape(r['spacy_top'])}</td>"
            f"<td>{html_lib.escape(r['sentence'])}</td>"
            "</tr>"
        )
    html_rows = "\n".join(html_rows)
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>Probe vs spaCy KL divergences</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem; text-align: left; }}
th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Probe vs spaCy (top {len(rows)} by KL)</h1>
<table>
<thead>
<tr>
<th>Dataset idx</th>
<th>Target token</th>
<th>Probe pred</th>
<th>KL divergence</th>
<th>Probe top probs</th>
<th>spaCy top probs</th>
<th>Sentence</th>
</tr>
</thead>
<tbody>
{html_rows}
</tbody>
</table>
</body>
</html>"""
    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML summary to {HTML_OUTPUT}")


def main():
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    num_written = 0
    num_skipped = 0
    html_rows = []

    with jsonl.open(input_path) as reader, jsonl.open(output_path, "w") as writer:
        metadata = None
        probe_classes = None
        tag_names = None

        for obj in reader:
            if metadata is None and isinstance(obj, dict) and "metadata" in obj:
                metadata = obj["metadata"]
                probe_classes = metadata.get("probe_classes")
                tag_names = metadata.get("tag_names")
                if tag_names is None:
                    raise ValueError("Metadata missing tag_names.")
                output_meta = {
                    "source": INPUT_PATH,
                    "epsilon": EPS,
                    "probe_classes": probe_classes,
                    "tag_names": tag_names,
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

            spacy_vec, spacy_count = vector_from_spacy(completions, tag_names)
            if spacy_count == 0:
                num_skipped += 1
                continue

            probe_vec = vector_from_probe(candidate, probe_classes, tag_names)
            kl = kl_divergence(probe_vec, spacy_vec)

            record = {
                "dataset_idx": candidate.get("dataset_idx"),
                "target_token": candidate.get("target_token_text"),
                "probe_pred": candidate.get("probe_pred_name"),
                "probe_distribution": dist_to_dict(probe_vec, tag_names),
                "spacy_distribution": dist_to_dict(spacy_vec, tag_names),
                "num_spacy_samples": spacy_count,
                "kl_divergence": kl,
                "sentence_text": candidate.get("sentence_text"),
            }
            writer.write(record)
            sentence = candidate.get("sentence_text") or ""
            html_rows.append(
                {
                    "dataset_idx": record["dataset_idx"],
                    "target_token": record["target_token"],
                    "probe_pred": record["probe_pred"],
                    "kl_divergence": kl,
                    "probe_top": top_tags(probe_vec, tag_names),
                    "spacy_top": top_tags(spacy_vec, tag_names),
                    "sentence": sentence,
                }
            )
            num_written += 1

    print(f"Wrote {num_written} summaries to {OUTPUT_PATH} (skipped {num_skipped}).")
    render_html(html_rows)


if __name__ == "__main__":
    main()
