#!/usr/bin/env python
"""Project component activations onto PCA planes and create grid plots."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from tqdm import tqdm

from coactivation_manifolds.component_graph import ComponentGraphConfig, compute_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA projections for coactivation components")
    parser.add_argument("run_dir", type=Path, help="Activation run directory containing activations/ and metadata/")
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
        "--device",
        default="cpu",
        help="Device for loading SAE when generating decoder directions (default: cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Batch size for streaming coactivations (default: 1e6)",
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
        "--min-size",
        type=int,
        default=2,
        help="Minimum component size to visualize (default: 2)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Limit number of components to plot (processed in descending size)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid dimension for plots per page (default: 10 => 100 plots)",
    )
    parser.add_argument(
        "--min-activations",
        type=int,
        default=16,
        help="Minimum activation rows required to run PCA for a component",
    )
    parser.add_argument(
        "--max-pc",
        type=int,
        default=6,
        help="Highest principal component to plot (e.g., 6 means plot up to PC6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PCA plot pages (defaults to metadata/component_pca)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    return parser.parse_args()


def collect_component_matrices(
    run_dir: Path,
    components: List[List[int]],
    *,
    first_token_idx: int,
    last_token_idx: int | None,
    show_progress: bool,
) -> List[np.ndarray]:
    if not components:
        return []

    activations_dir = run_dir / "activations"
    mapping: Dict[int, tuple[int, int]] = {}
    comp_lengths: List[int] = []
    for comp_idx, comp in enumerate(components):
        comp_lengths.append(len(comp))
        for local_idx, fid in enumerate(comp):
            mapping[int(fid)] = (comp_idx, local_idx)

    records: List[List[np.ndarray]] = [[] for _ in components]

    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)
    shard_iter = shard_paths
    if show_progress:
        shard_iter = tqdm(shard_paths, desc="Shards", leave=False)

    lower_bound = max(0, first_token_idx)

    for shard_path in shard_iter:
        file_path = shard_path / "data.parquet"
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(columns=["position_in_doc", "feature_ids", "activations"]):
            positions = batch.column("position_in_doc").to_numpy(zero_copy_only=False)
            feat_lists = batch.column("feature_ids").to_pylist()
            act_lists = batch.column("activations").to_pylist()
            for pos, feats, acts in zip(positions, feat_lists, act_lists):
                if pos < lower_bound:
                    continue
                if last_token_idx is not None and last_token_idx >= 0 and pos >= last_token_idx:
                    continue
                comp_hits: Dict[int, np.ndarray] = {}
                for fid, act in zip(feats, acts):
                    lookup = mapping.get(int(fid))
                    if lookup is None:
                        continue
                    comp_idx, local_idx = lookup
                    vec = comp_hits.get(comp_idx)
                    if vec is None:
                        vec = np.zeros(comp_lengths[comp_idx], dtype=np.float32)
                        comp_hits[comp_idx] = vec
                    vec[local_idx] = float(act)
                for comp_idx, vec in comp_hits.items():
                    if np.any(vec):
                        records[comp_idx].append(vec)

    matrices: List[np.ndarray] = []
    for comp_idx, rows in enumerate(records):
        if rows:
            matrices.append(np.vstack(rows))
        else:
            matrices.append(np.empty((0, comp_lengths[comp_idx]), dtype=np.float32))
    return matrices


def generate_plots(
    components: List[List[int]],
    matrices: List[np.ndarray],
    *,
    min_activations: int,
    max_pc: int,
    show_progress: bool,
) -> List[tuple[str, List[int], np.ndarray, np.ndarray]]:
    plots: List[tuple[str, List[int], np.ndarray, np.ndarray]] = []
    indices = range(len(components))
    if show_progress:
        indices = tqdm(indices, desc="Components", leave=False)

    for idx in indices:
        comp = components[idx]
        matrix = matrices[idx]
        if matrix.shape[0] < max(2, min_activations) or matrix.shape[1] < 2:
            continue
        n_components = min(matrix.shape)
        pca = PCA(n_components=min(n_components, max_pc))
        pcs = pca.fit_transform(matrix)
        usable_pcs = pcs.shape[1]
        for pc_idx in range(min(usable_pcs - 1, max_pc - 1)):
            x = pcs[:, pc_idx]
            y = pcs[:, pc_idx + 1]
            label = (
                f"Comp {idx} | size={len(comp)} | tok={matrix.shape[0]} | "
                f"PC{pc_idx+1}/PC{pc_idx+2}"
            )
            plots.append((label, comp, x, y))
    return plots


def write_plot_pages(
    plots: List[tuple[str, List[int], np.ndarray, np.ndarray]],
    output_dir: Path,
    grid_size: int,
    show_progress: bool,
) -> None:
    if not plots:
        print("No PCA plots generated")
        return

    plots_per_page = grid_size * grid_size
    total_pages = math.ceil(len(plots) / plots_per_page)

    page_iter = range(total_pages)
    if show_progress:
        page_iter = tqdm(page_iter, desc="Plot pages", leave=False)

    for page_idx in page_iter:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.2, grid_size * 2))
        axes_flat = axes.flatten()
        start = page_idx * plots_per_page
        end = min(start + plots_per_page, len(plots))
        for ax, plot_idx in zip(axes_flat, range(start, start + plots_per_page)):
            if plot_idx >= end:
                ax.axis("off")
                continue
            label, comp, x, y = plots[plot_idx]
            ax.scatter(x, y, s=6, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=6)
            fid_text = "\n".join(str(fid) for fid in comp)
            ax.text(
                1.04,
                0.5,
                fid_text,
                fontsize=5,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout(pad=0.4)
        output_path = output_dir / f"component_pca_{page_idx:03d}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    metadata_dir = run_dir / "metadata"
    coactivations_path = metadata_dir / "coactivations.parquet"
    feature_counts_path = metadata_dir / "feature_counts_trimmed.parquet"

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
        device=args.device,
        density_threshold=args.density_threshold,
        batch_size=args.batch_size,
    )

    result = compute_components(config)
    components = [comp for comp in result.components if len(comp) >= args.min_size]
    components = sorted(components, key=len, reverse=True)
    if args.max_components is not None:
        components = components[: args.max_components]

    if not components:
        print("No components of size >1 found under the given thresholds")
        return

    metadata_first = result.first_token_idx
    metadata_last = result.last_token_idx if result.last_token_idx >= 0 else None
    resolved_first = args.first_token_idx if args.first_token_idx is not None else metadata_first
    resolved_last = args.last_token_idx if args.last_token_idx is not None else metadata_last

    matrices = collect_component_matrices(
        run_dir,
        components,
        first_token_idx=resolved_first,
        last_token_idx=resolved_last,
        show_progress=not args.no_progress,
    )

    plots = generate_plots(
        components,
        matrices,
        min_activations=args.min_activations,
        max_pc=args.max_pc,
        show_progress=not args.no_progress,
    )

    output_dir = args.output_dir or (metadata_dir / "component_pca")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_plot_pages(plots, output_dir, args.grid_size, show_progress=not args.no_progress)


if __name__ == "__main__":
    main()
