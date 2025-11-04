#!/usr/bin/env python
"""Visualize SVD of transformer hidden states (not SAE activations) with projected decoder directions."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from coactivation_manifolds.component_graph import (
    ComponentGraphConfig,
    compute_components,
    _resolve_decoder_vectors,
)
from coactivation_manifolds.default_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER,
    NEURONPEDIA_BASE_URL,
)

GRID_COLUMNS = 4
GRID_ROWS = 2
PAGE_SIZE = GRID_COLUMNS * GRID_ROWS
FEATURE_BASE_URL = NEURONPEDIA_BASE_URL


@dataclass
class PCAResult:
    component_index: int
    feature_ids: List[int]
    pcs: np.ndarray
    variance_ratio: np.ndarray
    token_count: int
    activation_counts: np.ndarray
    count_hist: List[int]
    pca_components: np.ndarray
    pca_mean: np.ndarray
    snippets: List[str]
    n_pcs: int  # Number of PCs computed
    decoder_directions: np.ndarray  # [n_features, d_model] SAE decoder vectors
    highlight_mask: List[bool]

    @property
    def feature_count(self) -> int:
        return len(self.feature_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D SVD projections of transformer hidden states for coactivation clusters"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Activation run directory containing activations/ and metadata/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="monology/pile-uncopyrighted",
        help="Dataset name or local path (default: monology/pile-uncopyrighted)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field containing raw text (default: text)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=DEFAULT_LAYER,
        help=f"Model layer to extract hidden states from (default: {DEFAULT_LAYER})",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for model inference (default: cuda)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model inference (default: 4)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length (default: 1024)",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Minimum Jaccard similarity to keep edges (default: 0.8)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="Maximum cosine similarity to keep edges (optional)",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=None,
        help="Drop features with activation density above this fraction",
    )
    parser.add_argument(
        "--decoder-path",
        type=Path,
        default=None,
        help="Decoder matrix (.npy/.npz); cached automatically if omitted",
    )
    parser.add_argument(
        "--sae-release",
        default=None,
        help="sae-lens release identifier (used if decoder needs to be generated)",
    )
    parser.add_argument(
        "--sae-name",
        default=None,
        help="sae-lens SAE identifier within the release",
    )
    parser.add_argument(
        "--first-token-idx",
        type=int,
        default=None,
        help="Inclusive token position to start reading activations (default: metadata value)",
    )
    parser.add_argument(
        "--last-token-idx",
        type=int,
        default=None,
        help="Exclusive token position to stop reading activations (default: metadata value)",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=3,
        help="Feature count to target; defaults to 3",
    )
    parser.add_argument(
        "--include-larger",
        action="store_true",
        help="Include components with more than --min-features features",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Limit number of components to plot (processed in descending size)",
    )
    parser.add_argument(
        "--min-activations",
        type=int,
        default=16,
        help="Minimum activation rows required to run SVD for a component",
    )
    parser.add_argument(
        "--max-tokens-per-component",
        type=int,
        default=10000,
        help="Maximum tokens to use per component for SVD (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the dashboard (defaults to metadata/transformer_pca_3d)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--stream-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the dataset instead of downloading locally (default: True)",
    )
    return parser.parse_args()


def collect_token_metadata(
    run_dir: Path,
    components: List[List[int]],
    *,
    first_token_idx: int,
    last_token_idx: int | None,
    max_tokens_per_component: int,
    show_progress: bool,
    highlight_substring: str | None = None,
) -> tuple[List[List[Tuple[int, int, int]]], List[List[str]], List[List[bool]]]:
    """Collect (doc_id, position_in_doc, token_index) tuples plus snippet metadata.

    Returns:
        tokens_per_component: List of [(doc_id, position_in_doc, token_index), ...] per component
        snippets_per_component: List of [snippet_text, ...] per component
        highlight_masks: List of [bool, ...] indicating substring matches per token
    """
    if not components:
        return [], []

    activations_dir = run_dir / "activations"
    mapping: dict[int, int] = {}  # feature_id -> component_index
    for comp_idx, comp in enumerate(components):
        for fid in comp:
            mapping[int(fid)] = comp_idx

    tokens_per_component: List[List[Tuple[int, int, int]]] = [[] for _ in components]
    snippets_per_component: List[List[str]] = [[] for _ in components]
    highlight_masks: List[List[bool]] = [[] for _ in components]
    seen_tokens_per_component: List[set] = [set() for _ in components]

    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)
    shard_iter = shard_paths
    if show_progress:
        shard_iter = tqdm(shard_paths, desc="Scanning shards", leave=False)

    highlight_substring = highlight_substring.lower() if highlight_substring else None

    def _extract_center_token(snippet: str) -> str:
        if not snippet:
            return ""
        if "«" in snippet and "»" in snippet:
            try:
                return snippet.split("«", 1)[1].split("»", 1)[0]
            except IndexError:
                return snippet
        return snippet

    lower_bound = max(0, first_token_idx)

    for shard_path in shard_iter:
        file_path = shard_path / "data.parquet"
        pf = pq.ParquetFile(file_path)

        # Check for token_text
        sidecar_path = shard_path / "token_text.parquet"
        shard_snippets: Optional[List[str]] = None
        sidecar_offset = 0
        has_inline_snippets = "token_text" in pf.schema.names
        if not has_inline_snippets and sidecar_path.exists():
            shard_snippets = pq.read_table(sidecar_path, columns=["token_text"]).column(0).to_pylist()

        columns = ["doc_id", "position_in_doc", "token_index", "feature_ids"]
        if has_inline_snippets:
            columns.append("token_text")

        for batch in pf.iter_batches(columns=columns):
            doc_ids = batch.column("doc_id").to_numpy(zero_copy_only=False)
            positions = batch.column("position_in_doc").to_numpy(zero_copy_only=False)
            token_indices = batch.column("token_index").to_numpy(zero_copy_only=False)
            feat_lists = batch.column("feature_ids").to_pylist()

            if has_inline_snippets:
                text_list = batch.column("token_text").to_pylist()
            elif shard_snippets is not None:
                text_list = shard_snippets[sidecar_offset : sidecar_offset + len(positions)]
                sidecar_offset += len(positions)
            else:
                text_list = [""] * len(positions)

            for doc_id, pos, token_idx, feats, snippet_text in zip(
                doc_ids, positions, token_indices, feat_lists, text_list
            ):
                if pos < lower_bound:
                    continue
                if last_token_idx is not None and last_token_idx >= 0 and pos >= last_token_idx:
                    continue

                # Check which components this token belongs to
                for fid in feats:
                    comp_idx = mapping.get(int(fid))
                    if comp_idx is None:
                        continue

                    # Avoid duplicates and enforce max tokens
                    if token_idx in seen_tokens_per_component[comp_idx]:
                        continue
                    if len(tokens_per_component[comp_idx]) >= max_tokens_per_component:
                        continue

                    seen_tokens_per_component[comp_idx].add(token_idx)
                    tokens_per_component[comp_idx].append((int(doc_id), int(pos), int(token_idx)))
                    snippets_per_component[comp_idx].append(snippet_text)
                    if highlight_substring:
                        center_token = _extract_center_token(snippet_text).lower()
                        highlight_masks[comp_idx].append(
                            highlight_substring in center_token
                        )
                    else:
                        highlight_masks[comp_idx].append(False)

    return tokens_per_component, snippets_per_component, highlight_masks


def collect_transformer_states(
    tokens_per_component: List[List[Tuple[int, int, int]]],
    components: List[List[int]],
    *,
    dataset,
    text_field: str,
    model_name: str,
    layer_index: int,
    device: str,
    batch_size: int,
    max_length: int,
    show_progress: bool,
) -> List[np.ndarray]:
    """Re-run model to extract transformer hidden states for specified tokens.

    Returns:
        List of [n_tokens, d_model] arrays, one per component
    """
    # Load model and tokenizer
    if show_progress:
        print(f"Loading model {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    torch_dtype = torch.bfloat16 if "cuda" in device else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()

    # Organize tokens by doc_id
    doc_to_tokens: Dict[int, List[Tuple[int, int, int, int]]] = {}  # doc_id -> [(comp_idx, position, token_idx, local_idx), ...]
    for comp_idx, token_list in enumerate(tokens_per_component):
        for local_idx, (doc_id, position, token_idx) in enumerate(token_list):
            if doc_id not in doc_to_tokens:
                doc_to_tokens[doc_id] = []
            doc_to_tokens[doc_id].append((comp_idx, position, token_idx, local_idx))

    # Sort doc_ids for sequential access
    doc_ids_needed = sorted(doc_to_tokens.keys())

    # Initialize output matrices
    d_model = model.config.hidden_size
    matrices = [
        np.zeros((len(token_list), d_model), dtype=np.float32)
        for token_list in tokens_per_component
    ]

    # Stream through dataset and process needed documents
    doc_iterator = enumerate(dataset)
    if show_progress:
        print(f"Processing {len(doc_ids_needed)} documents...")
        doc_iterator = tqdm(enumerate(dataset), desc="Documents", total=len(doc_ids_needed), leave=False)

    docs_found = 0
    for doc_id, sample in doc_iterator:
        if doc_id not in doc_to_tokens:
            if docs_found >= len(doc_ids_needed):
                break
            continue

        docs_found += 1
        text = sample[text_field]

        # Tokenize
        tokenized = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Get hidden states
        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_index]  # [1, seq_len, d_model]

        # Convert to float32 first (numpy doesn't support bfloat16)
        hidden_np = hidden_states[0].float().cpu().numpy()  # [seq_len, d_model]

        # Map positions to hidden states
        for comp_idx, position, token_idx, local_idx in doc_to_tokens[doc_id]:
            if position < hidden_np.shape[0]:
                matrices[comp_idx][local_idx] = hidden_np[position]

    if show_progress:
        print(f"Collected hidden states for {docs_found} documents")

    return matrices


def generate_pca_results(
    components: List[List[int]],
    matrices: List[np.ndarray],
    snippets: List[List[str]],
    tokens_metadata: List[List[Tuple[int, int, int]]],
    highlight_masks: List[List[bool]],
    decoder_directions: np.ndarray,
    b_dec: np.ndarray,
    *,
    min_activations: int,
    show_progress: bool,
) -> List[PCAResult]:
    """Run SVD on SAE decoder directions and project activations onto this basis.

    The SVD basis is computed from decoder directions (without centering), so it represents
    the geometry of the SAE feature subspace relative to the origin. Activations are centered
    at b_dec and projected onto this feature-defined basis.

    Using SVD (not PCA) means we don't center the decoders at their mean. This makes the
    origin in the visualization space correspond to b_dec (the SAE's natural baseline).

    Args:
        matrices: List of [n_tokens, d_model] hidden state matrices
        decoder_directions: [n_total_features, d_model] SAE decoder matrix
        b_dec: [d_model] SAE decoder bias (natural origin for feature geometry)
        highlight_masks: Boolean flags indicating substring matches per token
    """
    results: List[PCAResult] = []
    indices = range(len(components))
    if show_progress:
        indices = tqdm(indices, desc="Computing SVD", leave=False)

    for idx in indices:
        comp = components[idx]
        matrix = matrices[idx]
        texts = snippets[idx] if idx < len(snippets) else []
        token_meta = tokens_metadata[idx]
        highlights = highlight_masks[idx] if idx < len(highlight_masks) else []

        if matrix.shape[0] < max(min_activations, 3):
            continue

        # Center activations at b_dec (SAE's natural origin)
        # The SAE decoders were learned relative to this baseline
        matrix_centered = matrix - b_dec

        # Count how many features are active per token (from metadata)
        activation_counts = np.ones(matrix.shape[0], dtype=np.int8)

        # Extract decoder directions for this component
        comp_decoders = np.array([decoder_directions[fid] for fid in comp], dtype=np.float32)

        # Compute SVD on decoder directions (without centering)
        # This defines the subspace in terms of SAE feature geometry relative to b_dec
        n_features = len(comp)
        d_model = matrix.shape[1]
        n_components = min(n_features, d_model)  # Limited by decoder matrix dims
        n_components = max(3, n_components)  # Need at least 3 for 3D plots

        try:
            # Use SVD (no centering) to find principal directions in decoder subspace
            U, S, Vt = np.linalg.svd(comp_decoders, full_matrices=False)
            basis_components = Vt[:n_components]  # [n_components, d_model]

            # Project centered activations (centered at b_dec) onto SVD basis
            pcs = matrix_centered @ basis_components.T

            # Compute variance explained from singular values
            total_var = np.sum(S**2)
            var = (S[:n_components]**2) / total_var if total_var > 0 else np.zeros(n_components, dtype=np.float32)
        except Exception:
            continue

        # Placeholder histogram (we don't track SAE activation counts in this version)
        count_hist = [0, 0, matrix.shape[0]]

        n_rows = pcs.shape[0]
        if len(texts) < n_rows:
            texts = texts + [""] * (n_rows - len(texts))
        else:
            texts = texts[:n_rows]

        if len(highlights) < n_rows:
            highlights = highlights + [False] * (n_rows - len(highlights))
        else:
            highlights = highlights[:n_rows]

        results.append(
            PCAResult(
                component_index=idx,
                feature_ids=list(comp),
                pcs=pcs,
                variance_ratio=var,
                token_count=matrix.shape[0],
                activation_counts=activation_counts,
                count_hist=count_hist,
                pca_components=basis_components.copy(),  # SVD components (no centering)
                pca_mean=np.zeros(d_model, dtype=np.float32),  # No centering, origin is b_dec
                snippets=texts,
                n_pcs=n_components,
                decoder_directions=comp_decoders,
                highlight_mask=highlights,
            )
        )
    return results


def generate_pc_triplets(n_pcs: int) -> List[tuple[int, int, int]]:
    """Generate consecutive PC triplets for 3D visualization.

    For n_pcs=5: returns [(0,1,2), (1,2,3), (2,3,4)]
    For n_pcs=3: returns [(0,1,2)]
    """
    if n_pcs < 3:
        return []
    triplets = []
    for i in range(n_pcs - 2):
        triplets.append((i, i + 1, i + 2))
    return triplets


def _make_figure_spec(entry: PCAResult, pc_x: int = 0, pc_y: int = 1, pc_z: int = 2) -> dict:
    """Generate a 3D plot spec for the specified component indices with decoder directions."""
    counts = entry.activation_counts.astype(int)
    snippets = entry.snippets if entry.snippets else []
    highlight_mask = entry.highlight_mask if entry.highlight_mask else [False] * len(counts)
    if len(highlight_mask) < len(counts):
        highlight_mask = highlight_mask + [False] * (len(counts) - len(highlight_mask))
    else:
        highlight_mask = highlight_mask[: len(counts)]

    customdata = [
        [
            int(counts[i]),
            snippets[i] if i < len(snippets) else "",
            "match" if highlight_mask[i] else "",
        ]
        for i in range(len(counts))
    ]

    colors = [
        "rgba(255,130,0,0.85)" if match else "rgba(0,0,255,0.1)"
        for match in highlight_mask
    ]
    sizes = [6 for _ in highlight_mask]

    pc_labels = [f"PC{pc_x+1}", f"PC{pc_y+1}", f"PC{pc_z+1}"]
    hovertemplate = (
        f"Comp {entry.component_index}<br>"
        f"{pc_labels[0]}: %{{x:.3f}}<br>{pc_labels[1]}: %{{y:.3f}}<br>{pc_labels[2]}: %{{z:.3f}}<br>"
        f"Features: {', '.join(str(fid) for fid in entry.feature_ids[:3])}<br>"
        "Text: %{customdata[1]}<br>"
        "Highlight: %{customdata[2]}<extra></extra>"
    )
    trace = go.Scatter3d(
        x=entry.pcs[:, pc_x],
        y=entry.pcs[:, pc_y],
        z=entry.pcs[:, pc_z],
        mode="markers",
        marker=dict(size=sizes, color=colors, line=dict(width=0)),
        customdata=customdata,
        name=f"Comp {entry.component_index}",
        hovertemplate=hovertemplate,
    )
    fig = go.Figure(data=[trace])

    # Project decoder directions onto the selected PC subspace
    selected_pcs = np.array([pc_x, pc_y, pc_z])
    pca_basis_3d = entry.pca_components[selected_pcs, :]  # [3, d_model]

    # Scale factor for visualization
    norms = np.linalg.norm(entry.pcs[:, selected_pcs], axis=1)
    target_length = float(np.percentile(norms, 90)) if norms.size else 1.0
    if not math.isfinite(target_length) or target_length <= 0:
        target_length = 1.0

    # Project each decoder direction
    for local_idx, fid in enumerate(entry.feature_ids):
        decoder_vec = entry.decoder_directions[local_idx]  # [d_model]

        # Normalize decoder (no centering needed with SVD)
        decoder_norm = np.linalg.norm(decoder_vec)
        if decoder_norm == 0:
            continue
        decoder_normalized = decoder_vec / decoder_norm

        # Project decoder onto SVD basis
        projection_3d = pca_basis_3d @ decoder_normalized  # [3]

        # Compute projection magnitude (fraction of vector in subspace)
        projection_magnitude = np.linalg.norm(projection_3d)

        # Scale for visualization
        scaled = projection_3d * target_length

        # Draw line from origin
        origin = np.zeros(3)
        fig.add_trace(
            go.Scatter3d(
                x=[0, float(scaled[0])],
                y=[0, float(scaled[1])],
                z=[0, float(scaled[2])],
                mode="lines+text",
                line=dict(color="red", width=4),
                text=["", f"{fid}<br>({projection_magnitude:.2f})"],
                textposition="top center",
                textfont=dict(size=10, color="black"),
                name=f"Feat {fid}",
                hovertemplate=f"Feature {fid}<br>Projection: {projection_magnitude:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=pc_labels[0], showticklabels=False, zeroline=True),
            yaxis=dict(title=pc_labels[1], showticklabels=False, zeroline=True),
            zaxis=dict(title=pc_labels[2], showticklabels=False, zeroline=True),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    spec = fig.to_dict()
    spec["config"] = {"responsive": True, "displayModeBar": False}
    return spec


def write_dashboard(
    results: List[PCAResult],
    output_path: Path,
    highlight_substring: str | None = None,
) -> None:
    """Generate HTML dashboard with interactive 3D SVD plots."""
    # Generate multiple figure specs per component (one per PC triplet)
    all_figure_specs = []
    components_payload = []

    for entry in results:
        triplets = generate_pc_triplets(entry.n_pcs)
        component_figures = []

        for pc_x, pc_y, pc_z in triplets:
            spec = _make_figure_spec(entry, pc_x, pc_y, pc_z)
            component_figures.append(spec)

        all_figure_specs.append(component_figures)

        components_payload.append({
            "component_index": entry.component_index,
            "token_count": entry.token_count,
            "feature_ids": entry.feature_ids,
            "variance": [float(v) for v in entry.variance_ratio],
            "count_hist": entry.count_hist,
            "n_pcs": entry.n_pcs,
            "pc_triplets": triplets,
            "highlight_count": int(sum(1 for flag in entry.highlight_mask if flag)),
            "highlight_total": len(entry.highlight_mask),
        })

    # Serialize figure specs as nested arrays
    figure_strings = [
        [json.dumps(spec, separators=(",", ":")) for spec in comp_figs]
        for comp_figs in all_figure_specs
    ]

    total_pages = max(1, math.ceil(len(results) / PAGE_SIZE))

    cell_markup: List[str] = []
    for cell_idx in range(PAGE_SIZE):
        cell_markup.append(
            (
                "        <div class=\"cell\" data-cell=\"{idx}\">\n"
                "          <div class=\"cell-info\" id=\"info-{idx}\"></div>\n"
                "          <div class=\"pc-triplet-nav\" id=\"pc-nav-{idx}\" style=\"display:none; margin-bottom: 0.3rem;\">\n"
                "            <button class=\"triplet-prev\" data-cell=\"{idx}\">◀</button>\n"
                "            <span class=\"triplet-indicator\" id=\"triplet-ind-{idx}\"></span>\n"
                "            <button class=\"triplet-next\" data-cell=\"{idx}\">▶</button>\n"
                "          </div>\n"
                "          <div class=\"plot-area\" id=\"plot-{idx}\"></div>\n"
                "          <div class=\"cell-actions\">\n"
                "            <button class=\"feature-button\" data-cell=\"{idx}\" data-slot=\"0\" data-feature=\"\">Feature 1</button>\n"
                "            <button class=\"feature-button\" data-cell=\"{idx}\" data-slot=\"1\" data-feature=\"\">Feature 2</button>\n"
                "            <button class=\"feature-button\" data-cell=\"{idx}\" data-slot=\"2\" data-feature=\"\">Feature 3</button>\n"
                "          </div>\n"
                "        </div>"
            ).format(idx=cell_idx)
        )
    cells_html = "\n".join(cell_markup)

    components_json = json.dumps(components_payload, separators=(",", ":"))
    figures_json = json.dumps(figure_strings, separators=(",", ":"))
    components_json_safe = components_json.replace("</", "<\\/")
    figures_json_safe = figures_json.replace("</", "<\\/")

    meta_note = (
        "SVD basis computed from SAE decoder directions (no centering). Origin = b_dec (SAE baseline). "
        "Highlighted tokens are tinted orange when a substring filter is provided. "
        "Red lines = decoder directions with projection magnitudes."
    )
    if highlight_substring:
        meta_note += f" Highlighting tokens containing \"{highlight_substring}\"."

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SAE Feature Subspace SVD Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 1.5rem; background: #f6f6f6; }}
    h1 {{ margin-top: 0; font-size: 1.4rem; }}
    .meta {{ margin: 0.5rem 0 1rem; color: #333; }}
    .nav {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }}
    .nav button {{ padding: 0.4rem 0.8rem; font-size: 0.8rem; cursor: pointer; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(240px, 1fr)); gap: 0.75rem; }}
    .cell {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.5rem; display: flex; flex-direction: column; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    .cell-info {{ font-size: 0.72rem; line-height: 1.3; margin-bottom: 0.5rem; color: #222; min-height: 4.8em; }}
    .plot-area {{ flex: 1 1 auto; min-height: 260px; border: 1px solid #eee; border-radius: 4px; background: #fafafa; }}
    .pc-triplet-nav {{ display: flex; align-items: center; gap: 0.4rem; font-size: 0.7rem; }}
    .pc-triplet-nav button {{ padding: 0.2rem 0.5rem; font-size: 0.7rem; cursor: pointer; }}
    .pc-triplet-nav button:disabled {{ cursor: not-allowed; opacity: 0.4; }}
    .triplet-indicator {{ font-weight: 500; color: #555; }}
    .cell-actions {{ margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }}
    .cell-actions button {{ padding: 0.3rem 0.6rem; font-size: 0.72rem; cursor: pointer; }}
    .cell-actions button:disabled {{ cursor: not-allowed; opacity: 0.45; }}
  </style>
</head>
<body>
  <h1>SAE Feature Subspace: SVD of Decoder Directions</h1>
  <div class="meta">Showing {PAGE_SIZE} plots per page (4 × 2 grid). Total components: {len(results)}. Use the arrows or ←/→ keys to switch pages. {meta_note}</div>
  <div class="nav">
    <button id="prev-page">◀ Prev</button>
    <span id="page-indicator"></span>
    <button id="next-page">Next ▶</button>
    <button id="unload-page">Unload this page</button>
  </div>
  <div class="grid">
{cells_html}
  </div>
  <script>
    const COMPONENTS = {components_json_safe};
    const FIGURE_STRINGS = {figures_json_safe};
    const PAGE_SIZE = {PAGE_SIZE};
    const TOTAL_COMPONENTS = COMPONENTS.length;
    const TOTAL_PAGES = Math.max(1, Math.ceil(TOTAL_COMPONENTS / PAGE_SIZE));
    const FEATURE_BASE_URL = "{FEATURE_BASE_URL}";

    const assignments = new Array(PAGE_SIZE).fill(null);
    const tripletIndices = new Array(PAGE_SIZE).fill(0);
    let currentPage = 0;

    function plotId(cell) {{ return 'plot-' + String(cell); }}
    function infoId(cell) {{ return 'info-' + String(cell); }}
    function pcNavId(cell) {{ return 'pc-nav-' + String(cell); }}
    function tripletIndId(cell) {{ return 'triplet-ind-' + String(cell); }}

    function cloneSpec(compIdx, tripletIdx) {{
      const figureArray = FIGURE_STRINGS[compIdx];
      if (!figureArray || tripletIdx >= figureArray.length) return null;
      return JSON.parse(figureArray[tripletIdx]);
    }}

    function purgeCell(cell) {{
      const container = document.getElementById(plotId(cell));
      if (!container) return;
      if (container.dataset.loaded === 'true') {{
        try {{ Plotly.purge(container); }} catch (err) {{ console.warn('Plotly purge failed', err); }}
        container.dataset.loaded = 'false';
        container.innerHTML = '';
      }}
    }}

    function unloadCurrentPage() {{
      for (let cell = 0; cell < PAGE_SIZE; cell += 1) {{
        purgeCell(cell);
      }}
    }}

    function formatCounts(hist) {{
      if (!Array.isArray(hist) || hist.length === 0) return '0/0/0';
      const padded = [hist[0] || 0, hist[1] || 0, hist[2] || 0];
      return padded.join('/');
    }}

    function formatComponent(metadata) {{
      const features = metadata.feature_ids.join(', ');
      const variance = metadata.variance.map(function(v) {{ return v.toFixed(2); }}).join(', ');
      const lines = [
        'Comp ' + metadata.component_index,
        'Tokens: ' + metadata.token_count,
        'Features (' + metadata.feature_ids.length + '): ' + features,
        'Variance: ' + variance,
      ];
      if (metadata.highlight_count && metadata.highlight_total) {{
        lines.push('Highlights: ' + metadata.highlight_count + ' / ' + metadata.highlight_total);
      }}
      return lines.join('<br>');
    }}

    function updateNav() {{
      const indicator = document.getElementById('page-indicator');
      if (indicator) {{
        indicator.textContent = 'Page ' + (currentPage + 1) + ' / ' + TOTAL_PAGES;
      }}
      const prev = document.getElementById('prev-page');
      const next = document.getElementById('next-page');
      if (prev) prev.disabled = currentPage <= 0;
      if (next) next.disabled = currentPage >= TOTAL_PAGES - 1;
    }}

    function updateTripletNav(cell) {{
      const globalIndex = assignments[cell];
      if (globalIndex === null || globalIndex >= TOTAL_COMPONENTS) return;

      const metadata = COMPONENTS[globalIndex];
      const numTriplets = metadata.pc_triplets ? metadata.pc_triplets.length : 1;
      const nav = document.getElementById(pcNavId(cell));

      if (numTriplets <= 1) {{
        if (nav) nav.style.display = 'none';
        return;
      }}

      if (nav) nav.style.display = 'flex';
      const indicator = document.getElementById(tripletIndId(cell));
      const tripletIdx = tripletIndices[cell] || 0;
      const triplet = metadata.pc_triplets[tripletIdx];
      if (indicator) {{
        const label = 'PC' + (triplet[0]+1) + '-' + (triplet[1]+1) + '-' + (triplet[2]+1) +
                      ' (' + (tripletIdx+1) + '/' + numTriplets + ')';
        indicator.textContent = label;
      }}

      const prevBtn = nav.querySelector('.triplet-prev');
      const nextBtn = nav.querySelector('.triplet-next');
      if (prevBtn) prevBtn.disabled = tripletIdx <= 0;
      if (nextBtn) nextBtn.disabled = tripletIdx >= numTriplets - 1;
    }}

    function assignCell(cell, globalIndex) {{
      assignments[cell] = globalIndex;
      tripletIndices[cell] = 0;
      const info = document.getElementById(infoId(cell));
      const buttons = document.querySelectorAll('.feature-button[data-cell="' + cell + '"]');
      const nav = document.getElementById(pcNavId(cell));
      purgeCell(cell);

      if (globalIndex === null || globalIndex >= TOTAL_COMPONENTS) {{
        if (info) info.innerHTML = '';
        if (nav) nav.style.display = 'none';
        buttons.forEach(function(button) {{
          button.disabled = true;
          button.dataset.feature = '';
          button.textContent = 'Feature';
        }});
        return;
      }}

      const metadata = COMPONENTS[globalIndex];
      if (info) info.innerHTML = formatComponent(metadata);

      buttons.forEach(function(button) {{
        const slot = parseInt(button.dataset.slot, 10) || 0;
        const featureId = metadata.feature_ids[slot] ?? null;
        if (featureId === null || featureId === undefined) {{
          button.disabled = true;
          button.dataset.feature = '';
          button.textContent = 'Feature';
        }} else {{
          button.disabled = false;
          button.dataset.feature = String(featureId);
          button.textContent = 'Feature ' + String(slot + 1) + ': ' + String(featureId);
        }}
      }});

      updateTripletNav(cell);
    }}

    function loadPlotByCell(cell, tripletIdx) {{
      const globalIndex = assignments[cell];
      if (globalIndex === null || globalIndex === undefined) return;
      if (tripletIdx === undefined) tripletIdx = tripletIndices[cell] || 0;

      const container = document.getElementById(plotId(cell));
      if (!container) return;

      purgeCell(cell);
      const spec = cloneSpec(globalIndex, tripletIdx);
      if (!spec) return;

      Plotly.newPlot(container, spec.data, spec.layout, spec.config || {{}});
      container.dataset.loaded = 'true';
      tripletIndices[cell] = tripletIdx;
      updateTripletNav(cell);
    }}

    function populatePage(page) {{
      currentPage = page;
      for (let cell = 0; cell < PAGE_SIZE; cell += 1) {{
        const globalIndex = page * PAGE_SIZE + cell;
        if (globalIndex < TOTAL_COMPONENTS) {{
          assignCell(cell, globalIndex);
          loadPlotByCell(cell);
        }} else {{
          assignCell(cell, null);
        }}
      }}
      updateNav();
    }}

    function gotoPage(delta) {{
      const target = Math.min(Math.max(currentPage + delta, 0), TOTAL_PAGES - 1);
      if (target === currentPage) return;
      unloadCurrentPage();
      populatePage(target);
    }}

    function setupControls() {{
      const prev = document.getElementById('prev-page');
      const next = document.getElementById('next-page');
      const unload = document.getElementById('unload-page');
      if (prev) prev.addEventListener('click', function() {{ gotoPage(-1); }});
      if (next) next.addEventListener('click', function() {{ gotoPage(1); }});
      if (unload) unload.addEventListener('click', unloadCurrentPage);

      document.querySelectorAll('.feature-button').forEach(function(button) {{
        button.addEventListener('click', function() {{
          const featureId = button.dataset.feature;
          if (!featureId) return;
          const url = FEATURE_BASE_URL + featureId;
          window.open(url, '_blank', 'noopener');
        }});
      }});

      document.querySelectorAll('.triplet-prev').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          const cell = parseInt(btn.dataset.cell, 10);
          const currentIdx = tripletIndices[cell] || 0;
          if (currentIdx > 0) {{
            loadPlotByCell(cell, currentIdx - 1);
          }}
        }});
      }});

      document.querySelectorAll('.triplet-next').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
          const cell = parseInt(btn.dataset.cell, 10);
          const globalIndex = assignments[cell];
          if (globalIndex === null) return;
          const metadata = COMPONENTS[globalIndex];
          const numTriplets = metadata.pc_triplets ? metadata.pc_triplets.length : 1;
          const currentIdx = tripletIndices[cell] || 0;
          if (currentIdx < numTriplets - 1) {{
            loadPlotByCell(cell, currentIdx + 1);
          }}
        }});
      }});

      document.addEventListener('keydown', function(event) {{
        if (event.key === 'ArrowLeft') {{
          gotoPage(-1);
        }} else if (event.key === 'ArrowRight') {{
          gotoPage(1);
        }}
      }});
    }}

    function init() {{
      setupControls();
      populatePage(0);
    }}

    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', init);
    }} else {{
      init();
    }}
  </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")
    print(
        f"Wrote {output_path} (components={len(results)}, pages={total_pages}, grid={GRID_COLUMNS}x{GRID_ROWS})"
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_dir = args.run_dir
    metadata_dir = run_dir / "metadata"
    coactivations_path = metadata_dir / "coactivations.parquet"
    feature_counts_path = metadata_dir / "feature_counts_trimmed.parquet"

    # Compute components
    config = ComponentGraphConfig(
        coactivations_path=coactivations_path,
        feature_counts_path=feature_counts_path,
        jaccard_threshold=args.jaccard_threshold,
        cosine_threshold=args.cosine_threshold,
        decoder_path=args.decoder_path,
        sae_release=args.sae_release
        or ComponentGraphConfig.__dataclass_fields__["sae_release"].default,
        sae_name=args.sae_name
        or ComponentGraphConfig.__dataclass_fields__["sae_name"].default,
        device="cpu",  # Just for decoder loading
        density_threshold=args.density_threshold,
        batch_size=1_000_000,
    )

    result = compute_components(config)
    if args.include_larger:
        components = [comp for comp in result.components if len(comp) >= args.min_features]
    else:
        components = [comp for comp in result.components if len(comp) == args.min_features]
    components = sorted(components, key=len, reverse=True)
    if args.max_components is not None:
        components = components[: args.max_components]

    if not components:
        if args.include_larger:
            print("No components meeting the minimum size were found")
        else:
            print("No components with the requested feature count were found")
        return

    # Load decoder directions - need FULL SAE feature count, not just max in components
    full_counts_path = metadata_dir / "feature_counts.parquet"
    if not full_counts_path.exists():
        raise FileNotFoundError(
            f"Missing feature_counts.parquet at {full_counts_path}. "
            "This file is needed to determine the SAE's total feature count."
        )
    full_counts_table = pq.read_table(full_counts_path)
    total_features = full_counts_table.num_rows

    decoder_directions = _resolve_decoder_vectors(
        feature_count=total_features,
        config=config,
        metadata_dir=metadata_dir,
    )

    # Load SAE to get b_dec (decoder bias, defines SAE's natural origin)
    print("Loading SAE to extract b_dec...")
    from coactivation_manifolds.sae_loader import load_sae
    sae_handle = load_sae(
        sae_release=config.sae_release,
        sae_name=config.sae_name,
        device="cpu"
    )
    b_dec = sae_handle.sae.b_dec.detach().cpu().numpy().astype(np.float32)  # [d_model]

    metadata_first = result.first_token_idx
    metadata_last = result.last_token_idx if result.last_token_idx >= 0 else None
    resolved_first = args.first_token_idx if args.first_token_idx is not None else metadata_first
    resolved_last = args.last_token_idx if args.last_token_idx is not None else metadata_last

    # Collect token metadata
    print("Collecting token metadata...")
    tokens_per_component, snippets_per_component, highlight_masks = collect_token_metadata(
        run_dir,
        components,
        first_token_idx=resolved_first,
        last_token_idx=resolved_last,
        max_tokens_per_component=args.max_tokens_per_component,
        show_progress=not args.no_progress,
    )

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.stream_dataset,
    )

    # Collect transformer hidden states
    matrices = collect_transformer_states(
        tokens_per_component,
        components,
        dataset=dataset,
        text_field=args.text_field,
        model_name=args.model_name,
        layer_index=args.layer_index,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        show_progress=not args.no_progress,
    )

    # Generate PCA results
    results = generate_pca_results(
        components,
        matrices,
        snippets_per_component,
        tokens_per_component,
        highlight_masks,
        decoder_directions,
        b_dec,
        min_activations=args.min_activations,
        show_progress=not args.no_progress,
    )

    if not results:
        print("No components had sufficient activations for SVD")
        return

    output_dir = args.output_dir or (metadata_dir / "transformer_pca_3d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transformer_pca_dashboard.html"
    write_dashboard(results, output_path)


if __name__ == "__main__":
    main()
