from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import jsonlines

# Configuration
CACHE_BASE = Path("activation_cache/distill_cache_train_pca")  # base path to *_Y.npy and *_meta.npz
JSONL_PATH = Path("data/merged_shards.jsonl")  # fallback if cache not used
USE_CACHE = True
PRESENCE_THRESH = 0.0  # label considered present if prob > thresh
OUTPUT_PNG = Path("models/jaccard_heatmap.png")
TOP_TRIPLETS = 10  # number of top triplets to print


def load_from_cache():
    y_path = CACHE_BASE.with_name(CACHE_BASE.name + "_Y.npy")
    meta_path = CACHE_BASE.with_name(CACHE_BASE.name + "_meta.npz")
    if not (y_path.exists() and meta_path.exists()):
        return None, None
    Y = np.load(y_path)
    meta = np.load(meta_path, allow_pickle=True)
    tags = meta.get("tag_names")
    if tags is None:
        return None, None
    return Y, list(tags)


def load_from_jsonl():
    tag_set = set()
    rows: List[List[str]] = []
    with jsonlines.open(JSONL_PATH) as reader:
        for obj in reader:
            if isinstance(obj, dict) and "metadata" in obj:
                meta = obj["metadata"]
                tags = meta.get("tag_names")
                if tags:
                    tag_set.update(tags)
                continue
            present = set()
            for comp in obj.get("completions", []):
                pos = comp.get("spacy_pos")
                if pos:
                    present.add(pos)
                    tag_set.add(pos)
            if present:
                rows.append(list(present))
    tag_names = sorted(tag_set)
    tag_idx = {t: i for i, t in enumerate(tag_names)}
    Y = np.zeros((len(rows), len(tag_names)), dtype=np.float32)
    for i, tags in enumerate(rows):
        for t in tags:
            Y[i, tag_idx[t]] = 1.0
    return Y, tag_names


def compute_jaccard(Y: np.ndarray):
    # Binarize
    B = (Y > PRESENCE_THRESH).astype(np.float32)
    # Per-tag counts
    counts = B.sum(axis=0)  # shape (C,)
    # Intersection matrix
    intersect = B.T @ B  # (C,C)
    # Union = count_i + count_j - intersect
    union = counts[:, None] + counts[None, :] - intersect
    # Avoid div by zero
    union = np.where(union == 0, 1.0, union)
    jacc = intersect / union
    return jacc


def top_triplets(Y: np.ndarray, tags: List[str], top_n: int = 10):
    B = (Y > PRESENCE_THRESH)
    n_tags = len(tags)
    counts = B.sum(axis=0)
    N = B.shape[0]
    triples = []
    for i in range(n_tags):
        for j in range(i + 1, n_tags):
            pair = B[:, i] & B[:, j]
            if not pair.any():
                continue
            for k in range(j + 1, n_tags):
                c = int((pair & B[:, k]).sum())
                if c == 0:
                    continue
                # lift-style normalization against independence
                expected = (counts[i] * counts[j] * counts[k]) / (N ** 2) if N > 0 else 1
                lift = c / expected if expected > 0 else 0
                triples.append((lift, c, (tags[i], tags[j], tags[k])))
    triples.sort(reverse=True, key=lambda x: x[0])
    return triples[:top_n]


def plot_heatmap(jacc: np.ndarray, tags: List[str], out_path: Path):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(jacc, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(tags)), tags, rotation=90, fontsize=6)
    plt.yticks(range(len(tags)), tags, fontsize=6)
    plt.title("Jaccard similarity of tag co-occurrence")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved Jaccard heatmap to {out_path}")


def main():
    if USE_CACHE:
        Y, tags = load_from_cache()
    else:
        Y, tags = None, None
    if Y is None or tags is None:
        print("Cache unavailable or missing tags; falling back to JSONL")
        Y, tags = load_from_jsonl()
    if Y is None or tags is None:
        raise ValueError("Could not load labels.")
    jacc = compute_jaccard(Y)
    plot_heatmap(jacc, tags, OUTPUT_PNG)
    if TOP_TRIPLETS and TOP_TRIPLETS > 0:
        triples = top_triplets(Y, tags, TOP_TRIPLETS)
        print(f"Top {TOP_TRIPLETS} triplets by co-occurrence count:")
        for lift, c, trip in triples:
            print(f"{trip}: count={c}, lift={lift:.3f}")


if __name__ == "__main__":
    main()
