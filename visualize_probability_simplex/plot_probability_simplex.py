"""Visualize definition probabilities against PCA projections."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
import textwrap
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coactivation_manifolds.component_graph import ComponentGraphConfig, _resolve_decoder_vectors
from coactivation_manifolds.sae_loader import load_sae

DEFAULT_SCATTER_NAME = "{}_pca_scatter.png"
DEFAULT_ENTROPY_NAME = "{}_entropy_scatter.png"
DEFAULT_LEGEND_CHARS = 80
DEFAULT_MODEL_NAME = "google/gemma-2-2b"
DEFAULT_LAYER_INDEX = 12
DEFAULT_DATASET = "monology/pile-uncopyrighted"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res"
DEFAULT_SAE_NAME = "layer_12/width_65k/average_l0_72"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA of feature activations colored by definition probs")
    parser.add_argument("scored_path", type=Path, help="Parquet produced by score_definitions.py")
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=None,
        help="PNG output path for the colored scatter (default: <input>_pca_scatter.png)",
    )
    parser.add_argument("--point-size", type=float, default=30.0, help="Marker size for scatter plot")
    parser.add_argument("--alpha", type=float, default=0.55, help="Marker opacity")
    parser.add_argument(
        "--entropy-output",
        type=Path,
        default=None,
        help="PNG output path for entropy-colored scatter (default: <input>_entropy_scatter.png)",
    )
    parser.add_argument(
        "--legend-max-chars",
        type=int,
        default=DEFAULT_LEGEND_CHARS,
        help="Max characters per legend entry before truncation (default: 80)",
    )
    parser.add_argument("--model-name", default=None, help="Transformer model to re-run for hidden states (default: metadata or google/gemma-2-2b)")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=None,
        help="Transformer layer to sample residual stream activations from (default: metadata or 12)",
    )
    parser.add_argument("--device", default="cuda", help="Device for transformer inference (default: cuda)")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokenizer max_length when re-encoding documents (default: metadata or 1024)",
    )
    parser.add_argument("--dataset", default=None, help="Override dataset name (default: metadata)")
    parser.add_argument("--dataset-config", default=None, help="Override dataset config (default: metadata)")
    parser.add_argument("--dataset-split", default=None, help="Override dataset split (default: metadata)")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Optional `load_from_disk` dataset path override")
    parser.add_argument("--text-field", default=None, help="Override text field name (default: metadata or 'text')")
    parser.add_argument("--sae-release", default=None, help="SAE release for decoder/bias (default: gemma-scope-2b-pt-res)")
    parser.add_argument(
        "--sae-name",
        default=None,
        help="SAE identifier within release (default: layer_12/width_65k/average_l0_72)",
    )
    parser.add_argument("--sae-device", default="cpu", help="Device to load SAE on (default: cpu)")
    return parser.parse_args()


def default_paths(base: Path) -> Tuple[Path, Path]:
    stem = base.stem
    parent = base.parent
    return (
        parent / DEFAULT_SCATTER_NAME.format(stem),
        parent / DEFAULT_ENTROPY_NAME.format(stem),
    )


def load_metadata(table) -> Dict[str, str]:
    metadata = table.schema.metadata or {}
    return {key.decode(): value.decode() for key, value in metadata.items()}


def resolve_run_dir(metadata: Dict[str, str], scored_path: Path) -> Path:
    raw = metadata.get("run_dir")
    if not raw:
        raise ValueError("Parquet metadata missing 'run_dir'; recreate snippets with newer script.")
    path = Path(raw)
    if not path.is_absolute():
        path = (scored_path.parent / path).resolve()
    return path


def parse_feature_ids(metadata: Dict[str, str]) -> List[int]:
    raw = metadata.get("feature_ids")
    if not raw:
        raise ValueError("Parquet metadata missing 'feature_ids'; rerun collect_word_snippets.")
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Unable to parse feature_ids metadata") from exc
    return [int(v) for v in values]


def resolve_dataset_spec(args: argparse.Namespace, metadata: Dict[str, str], scored_path: Path) -> dict:
    resolved_path = None
    if args.dataset_path is not None:
        resolved_path = args.dataset_path.resolve()
    else:
        dataset_path_str = (metadata.get("dataset_path") or "").strip()
        if dataset_path_str:
            candidate = Path(dataset_path_str)
            if not candidate.is_absolute():
                candidate = (scored_path.parent / candidate).resolve()
            resolved_path = candidate

    dataset_name = args.dataset or metadata.get("dataset") or DEFAULT_DATASET
    dataset_config = args.dataset_config or metadata.get("dataset_config") or None
    dataset_split = args.dataset_split or metadata.get("dataset_split") or DEFAULT_DATASET_SPLIT
    text_field = args.text_field or metadata.get("text_field") or DEFAULT_TEXT_FIELD

    max_length_meta = metadata.get("tokenizer_max_length")
    if args.max_length is not None:
        max_length = args.max_length
    elif max_length_meta:
        max_length = int(max_length_meta)
    else:
        max_length = DEFAULT_MAX_LENGTH

    streaming_meta = metadata.get("dataset_streaming", "1")
    stream = streaming_meta != "0" and resolved_path is None

    return {
        "path": resolved_path,
        "dataset": dataset_name,
        "config": dataset_config or None,
        "split": dataset_split,
        "text_field": text_field,
        "max_length": max_length,
        "stream": stream,
    }


def load_dataset_handle(spec: dict):
    if spec["path"]:
        ds = load_from_disk(str(spec["path"]))
        return ds[spec["split"]], False
    return (
        load_dataset(
            spec["dataset"],
            spec["config"],
            split=spec["split"],
            streaming=spec["stream"],
        ),
        spec["stream"],
    )


def load_decoder_and_bias(
    run_dir: Path,
    sae_release: str,
    sae_name: str,
    sae_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    def _canonical_release(name: str) -> str:
        if not name:
            return name
        if "/" in name:
            # If someone stored "google/<repo>" we only need repo name for sae-lens releases.
            parts = name.split("/")
            # Some metadata may have accidentally stored "google/<repo>"; prefer the last chunk.
            return parts[-1]
        return name

    sae_release = _canonical_release(sae_release)

    metadata_dir = run_dir / "metadata"
    counts_path = metadata_dir / "feature_counts.parquet"
    if not counts_path.exists():
        raise FileNotFoundError(f"Missing feature_counts.parquet under {metadata_dir}")
    total_features = pq.read_table(counts_path, columns=["feature_id"]).num_rows

    config = ComponentGraphConfig(
        coactivations_path=metadata_dir / "coactivations.parquet",
        feature_counts_path=metadata_dir / "feature_counts_trimmed.parquet",
        decoder_path=metadata_dir / "decoder_directions.npy",
        sae_release=sae_release,
        sae_name=sae_name,
        device="cpu",
    )
    decoder_vectors = _resolve_decoder_vectors(
        feature_count=total_features,
        config=config,
        metadata_dir=metadata_dir,
    )

    sae_handle = load_sae(sae_release=sae_release, sae_name=sae_name, device=sae_device)
    b_dec = sae_handle.sae.b_dec.detach().cpu().numpy().astype(np.float32)
    return decoder_vectors.astype(np.float32, copy=False), b_dec


def load_model_and_tokenizer(model_name: str, device: str):
    resolved_device = device
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU for transformer inference.")
        resolved_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if resolved_device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(resolved_device)
    model.eval()
    return tokenizer, model


def collect_residual_states(
    table,
    dataset,
    text_field: str,
    tokenizer,
    model,
    layer_index: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    doc_ids = table.column("doc_id").to_pylist()
    positions = table.column("position_in_doc").to_pylist()

    doc_to_rows: Dict[int, List[tuple[int, int]]] = {}
    for row_idx, (doc_id, pos) in enumerate(zip(doc_ids, positions)):
        doc = int(doc_id)
        doc_to_rows.setdefault(doc, []).append((row_idx, int(pos)))

    d_model = model.config.hidden_size
    hidden = np.zeros((len(doc_ids), d_model), dtype=np.float32)
    filled = np.zeros(len(doc_ids), dtype=bool)

    docs_needed = len(doc_to_rows)
    seen_docs: set[int] = set()
    for doc_idx, sample in enumerate(dataset):
        if doc_idx not in doc_to_rows:
            continue
        text = sample.get(text_field)
        if text is None:
            continue

        tokenized = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)
        if layer_index >= len(outputs.hidden_states):
            raise ValueError(f"Layer index {layer_index} out of bounds for model with {len(outputs.hidden_states)} hidden states.")
        hidden_np = outputs.hidden_states[layer_index][0].float().cpu().numpy()

        for row_idx, pos in doc_to_rows[doc_idx]:
            if 0 <= pos < hidden_np.shape[0]:
                hidden[row_idx] = hidden_np[pos]
                filled[row_idx] = True

        seen_docs.add(doc_idx)
        if len(seen_docs) >= docs_needed:
            break

    missing_docs = sorted(set(doc_to_rows) - seen_docs)
    if missing_docs:
        raise RuntimeError(
            f"Dataset ended before all doc_ids were processed. Missing doc IDs: {missing_docs[:5]}{'...' if len(missing_docs) > 5 else ''}"
        )

    if not filled.any():
        raise RuntimeError("Failed to collect any residual activations for the requested tokens.")

    return hidden[filled], filled


def compute_decoder_basis(decoder_subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if decoder_subset.shape[0] < 3:
        raise ValueError("Need at least three feature directions to compute PCA basis.")
    n_components = min(decoder_subset.shape[0], decoder_subset.shape[1])
    U, S, Vt = np.linalg.svd(decoder_subset, full_matrices=False)
    basis = Vt[:n_components]
    total = np.sum(S**2)
    variance = (S[:n_components] ** 2) / total if total > 0 else np.zeros(n_components, dtype=np.float32)
    return basis.astype(np.float32), variance.astype(np.float32)


def regression_r2(pcs: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if pcs.shape[1] < 2:
        raise ValueError("Need at least two principal components for regression analysis")
    X = pcs[:, :2]
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    design = np.concatenate([X, ones], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(design, probs, rcond=None)
    predictions = design @ coeffs
    residual = np.sum((probs - predictions) ** 2, axis=0)
    total = np.sum((probs - probs.mean(axis=0, keepdims=True)) ** 2, axis=0)
    overall_total = total.sum()
    overall_residual = residual.sum()

    def safe_ratio(num: float, denom: float) -> float:
        return float("nan") if denom <= 0 else 1.0 - num / denom

    r2_per = [safe_ratio(residual[i], total[i]) for i in range(probs.shape[1])]
    overall = safe_ratio(overall_residual, overall_total)
    return {
        "overall": overall,
        "per_label": r2_per,
    }


def _base_palette() -> np.ndarray:
    return np.array(
        [
            (1.0, 0.2, 0.2),
            (0.0, 0.6, 0.0),
            (0.2, 0.2, 1.0),
        ],
        dtype=np.float32,
    )


def _compute_bounds(pcs: np.ndarray, margin_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    x = pcs[:, 0]
    y = pcs[:, 1]
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin_x = margin_ratio * (x_range if x_range > 0 else 1.0)
    margin_y = margin_ratio * (y_range if y_range > 0 else 1.0)
    return x.min() - margin_x, x.max() + margin_x, y.min() - margin_y, y.max() + margin_y


def _truncate_definition(text: str, max_chars: int) -> str:
    """Clamp long definition strings so legends don't squeeze the plot."""
    text = " ".join((text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    # textwrap.shorten enforces the placeholder inside width budget.
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Return normalized entropy (0-1) per row in bits."""
    eps = 1e-8
    safe = np.clip(probs, eps, 1.0)
    entropy = -np.sum(safe * np.log2(safe), axis=1)
    max_entropy = np.log2(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    if max_entropy <= 0:
        return entropy
    return entropy / max_entropy


def plot_definition_scatter(
    pcs: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    label_texts: List[str],
    output_path: Path,
    point_size: float,
    alpha: float,
    variance_ratio: np.ndarray,
    legend_max_chars: int,
) -> None:
    base_colors = _base_palette()
    colors = np.clip(probs @ base_colors, 0.0, 1.0)
    x_min, x_max, y_min, y_max = _compute_bounds(pcs)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=point_size, alpha=alpha, edgecolors="none")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.2%} var)" if variance_ratio.size > 1 else "PC2")
    ax.set_title("Definition probability simplex – scatter")
    ax.set_aspect('auto')
    handles = []
    for idx, (label, color) in enumerate(zip(labels[:3], base_colors[: len(labels)])):
        full_text = label_texts[idx] if idx < len(label_texts) else ""
        legend_label = f"{label}: {_truncate_definition(full_text, legend_max_chars)}"
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=legend_label, markerfacecolor=color, markersize=8)
        )
    if handles:
        legend = fig.legend(
            handles=handles,
            title="Definition",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
            borderaxespad=0.0,
            ncol=1,
        )
        with contextlib.suppress(AttributeError):
            legend._legend_box.align = "left"
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92 if handles else 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_entropy_scatter(
    pcs: np.ndarray,
    entropies: np.ndarray,
    output_path: Path,
    point_size: float,
    alpha: float,
    variance_ratio: np.ndarray,
) -> None:
    x_min, x_max, y_min, y_max = _compute_bounds(pcs)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = ax.scatter(
        pcs[:, 0],
        pcs[:, 1],
        c=entropies,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(f"PC1 ({variance_ratio[0]:.2%} var)")
    ax.set_ylabel(f"PC2 ({variance_ratio[1]:.2%} var)" if variance_ratio.size > 1 else "PC2")
    ax.set_title("Definition probability entropy")
    ax.set_aspect('auto')
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Normalized entropy (0‑1)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.scored_path)
    metadata = load_metadata(table)

    if "top_definition_probabilities" in table.column_names:
        probabilities = table.column("top_definition_probabilities").to_pylist()
    else:
        probabilities = table.column("definition_probabilities").to_pylist()
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape[1] < 3:
        raise ValueError("Need at least three definition probabilities to plot the simplex")

    run_dir = resolve_run_dir(metadata, args.scored_path)
    feature_ids = parse_feature_ids(metadata)

    labels = json.loads(metadata.get("top_definition_indices") or metadata.get("definition_labels", "[]"))
    if len(labels) != probs.shape[1]:
        labels = [f"Choice {i}" for i in range(probs.shape[1])]

    raw_texts = metadata.get("top_definition_texts") or metadata.get("definition_texts", "{}")
    try:
        parsed_texts = json.loads(raw_texts)
    except Exception:
        parsed_texts = {}

    if isinstance(parsed_texts, list):
        # Align list entries with labels; pad if needed
        label_texts = [(parsed_texts[i] if i < len(parsed_texts) else "") for i in range(len(labels))]
    elif isinstance(parsed_texts, dict):
        label_texts = [parsed_texts.get(label, "") for label in labels]
    else:
        label_texts = ["" for _ in labels]

    dataset_spec = resolve_dataset_spec(args, metadata, args.scored_path)
    dataset_handle, _ = load_dataset_handle(dataset_spec)

    model_name = args.model_name or metadata.get("activation_model") or DEFAULT_MODEL_NAME
    layer_index = args.layer_index if args.layer_index is not None else DEFAULT_LAYER_INDEX

    tokenizer, model = load_model_and_tokenizer(model_name, args.device)
    hidden_states, mask = collect_residual_states(
        table,
        dataset_handle,
        dataset_spec["text_field"],
        tokenizer,
        model,
        layer_index,
        dataset_spec["max_length"],
    )
    if mask is not None and mask.shape[0] == probs.shape[0] and not mask.all():
        kept = int(mask.sum())
        dropped = mask.shape[0] - kept
        probs = probs[mask]
        print(f"Dropped {dropped} snippet(s) without matching residual activations; {kept} remain.")

    sae_release = args.sae_release or metadata.get("sae_release") or DEFAULT_SAE_RELEASE
    sae_name = args.sae_name or metadata.get("sae_name") or DEFAULT_SAE_NAME
    decoder_vectors, b_dec = load_decoder_and_bias(run_dir, sae_release, sae_name, args.sae_device)

    max_feature_id = decoder_vectors.shape[0]
    if any(fid >= max_feature_id or fid < 0 for fid in feature_ids):
        raise ValueError("Feature IDs exceed available decoder directions; verify run_dir and SAE match.")
    decoder_subset = decoder_vectors[feature_ids]
    basis, variance_ratio = compute_decoder_basis(decoder_subset)
    pcs = (hidden_states - b_dec) @ basis.T
    if pcs.shape[0] < 2:
        raise ValueError("Need at least two valid activations to plot.")

    stats = regression_r2(pcs, probs)

    scatter_path, entropy_default = default_paths(args.scored_path)
    scatter_output = args.scatter_output or scatter_path
    entropy_output = args.entropy_output or entropy_default

    plot_definition_scatter(
        pcs,
        probs,
        labels,
        label_texts,
        scatter_output,
        args.point_size,
        args.alpha,
        variance_ratio,
        args.legend_max_chars,
    )
    print(f"Saved PCA scatter to {scatter_output}")

    entropies = compute_entropy(probs)
    plot_entropy_scatter(
        pcs,
        entropies,
        entropy_output,
        args.point_size,
        args.alpha,
        variance_ratio,
    )
    print(f"Saved entropy scatter to {entropy_output}")

    per_label_r2 = {label: value for label, value in zip(labels, stats["per_label"]) }
    print(f"Overall R^2 (PC1/PC2 -> probabilities): {stats['overall']:.4f}")
    for label in labels:
        value = per_label_r2.get(label, float("nan"))
        print(f"  {label}: {value:.4f}")


if __name__ == "__main__":
    main()
