#!/usr/bin/env python
"""Project multi-feature clusters onto 3D PCA space and export a Plotly HTML dashboard."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import plotly.graph_objects as go
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from tqdm import tqdm

from coactivation_manifolds.component_graph import ComponentGraphConfig, compute_components

GRID_COLUMNS = 4
GRID_ROWS = 2
PAGE_SIZE = GRID_COLUMNS * GRID_ROWS
FEATURE_BASE_URL = "https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-65k/"


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

    @property
    def feature_count(self) -> int:
        return len(self.feature_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D PCA projections for multi-feature coactivation clusters"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Activation run directory containing activations/ and metadata/",
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
        help="Minimum activation rows required to run PCA for a component",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the dashboard (defaults to metadata/component_pca_3d)",
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
) -> tuple[List[np.ndarray], List[List[str]]]:
    if not components:
        return [], []

    activations_dir = run_dir / "activations"
    mapping: dict[int, tuple[int, int]] = {}
    comp_lengths: List[int] = []
    for comp_idx, comp in enumerate(components):
        comp_lengths.append(len(comp))
        for local_idx, fid in enumerate(comp):
            mapping[int(fid)] = (comp_idx, local_idx)

    records: List[List[np.ndarray]] = [[] for _ in components]
    snippet_storage: List[List[str]] = [[] for _ in components]

    shard_paths = sorted(activations_dir.glob("shard=*"), key=lambda p: p.name)
    shard_iter = shard_paths
    if show_progress:
        shard_iter = tqdm(shard_paths, desc="Shards", leave=False)

    lower_bound = max(0, first_token_idx)

    for shard_path in shard_iter:
        file_path = shard_path / "data.parquet"
        pf = pq.ParquetFile(file_path)
        sidecar_path = shard_path / "token_text.parquet"
        shard_snippets: Optional[List[str]] = None
        sidecar_offset = 0
        has_inline_snippets = "token_text" in pf.schema.names
        if not has_inline_snippets and sidecar_path.exists():
            shard_snippets = pq.read_table(sidecar_path, columns=["token_text"]).column(0).to_pylist()
        columns = ["position_in_doc", "feature_ids", "activations"]
        if has_inline_snippets:
            columns.append("token_text")
        for batch in pf.iter_batches(columns=columns):
            positions = batch.column("position_in_doc").to_numpy(zero_copy_only=False)
            feat_lists = batch.column("feature_ids").to_pylist()
            act_lists = batch.column("activations").to_pylist()
            if has_inline_snippets:
                text_list = batch.column("token_text").to_pylist()
            elif shard_snippets is not None:
                text_list = shard_snippets[sidecar_offset : sidecar_offset + len(positions)]
                sidecar_offset += len(positions)
            else:
                text_list = [""] * len(positions)
            for pos, feats, acts, snippet_text in zip(positions, feat_lists, act_lists, text_list):
                if pos < lower_bound:
                    continue
                if last_token_idx is not None and last_token_idx >= 0 and pos >= last_token_idx:
                    continue
                comp_hits: dict[int, np.ndarray] = {}
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
                        snippet_storage[comp_idx].append(snippet_text)

    matrices: List[np.ndarray] = []
    snippets: List[List[str]] = []
    for comp_idx, rows in enumerate(records):
        if rows:
            matrices.append(np.vstack(rows))
            snippets.append(snippet_storage[comp_idx])
        else:
            matrices.append(np.empty((0, comp_lengths[comp_idx]), dtype=np.float32))
            snippets.append([])
    return matrices, snippets


def generate_pca_results(
    components: List[List[int]],
    matrices: List[np.ndarray],
    snippets: List[List[str]],
    *,
    min_activations: int,
    show_progress: bool,
) -> List[PCAResult]:
    results: List[PCAResult] = []
    indices = range(len(components))
    if show_progress:
        indices = tqdm(indices, desc="Components", leave=False)

    for idx in indices:
        comp = components[idx]
        matrix = matrices[idx]
        texts = snippets[idx] if idx < len(snippets) else []
        if matrix.shape[1] < 3:
            continue
        if matrix.shape[0] < max(min_activations, 3):
            continue
        activation_counts = np.count_nonzero(matrix > 0.0, axis=1).astype(np.int8)
        if activation_counts.size == 0:
            continue
        try:
            pca = PCA(n_components=3)
            pcs = pca.fit_transform(matrix)
        except Exception:
            continue
        var = pca.explained_variance_ratio_
        hist = np.bincount(activation_counts)
        one = int(hist[1]) if hist.size > 1 else 0
        two = int(hist[2]) if hist.size > 2 else 0
        three_plus = int(hist[3:].sum()) if hist.size > 3 else 0
        count_hist = [one, two, three_plus]
        if len(texts) < pcs.shape[0]:
            texts = texts + [""] * (pcs.shape[0] - len(texts))
        else:
            texts = texts[: pcs.shape[0]]
        results.append(
            PCAResult(
                component_index=idx,
                feature_ids=list(comp),
                pcs=pcs,
                variance_ratio=var,
                token_count=matrix.shape[0],
                activation_counts=activation_counts,
                count_hist=count_hist,
                pca_components=pca.components_.copy(),
                pca_mean=pca.mean_.copy(),
                snippets=texts,
            )
        )
    return results


def _make_figure_spec(entry: PCAResult) -> dict:
    counts = entry.activation_counts.astype(int)
    snippets = entry.snippets if entry.snippets else []
    customdata = [
        [int(counts[i]), snippets[i] if i < len(snippets) else ""]
        for i in range(len(counts))
    ]
    colors = []
    for raw_count in counts:
        count = int(raw_count)
        if count <= 0:
            colors.append("#777777")
        elif count == 1:
            colors.append("red")
        elif count == 2:
            colors.append("orange")
        else:
            colors.append("blue")
    hovertemplate = (
        f"Comp {entry.component_index}<br>"
        "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<br>"
        f"Features: {', '.join(str(fid) for fid in entry.feature_ids[:3])}" "<br>"
        "Active features: %{customdata[0]}<br>"
        "Text: %{customdata[1]}<extra></extra>"
    )
    trace = go.Scatter3d(
        x=entry.pcs[:, 0],
        y=entry.pcs[:, 1],
        z=entry.pcs[:, 2],
        mode="markers",
        marker=dict(size=3, opacity=0.65, color=colors),
        customdata=customdata,
        name=f"Comp {entry.component_index}",
        hovertemplate=hovertemplate,
    )
    fig = go.Figure(data=[trace])

    components_t = entry.pca_components.T
    baseline = -entry.pca_mean @ components_t
    norms = np.linalg.norm(entry.pcs, axis=1)
    target_length = float(np.percentile(norms, 90)) if norms.size else 1.0
    if not math.isfinite(target_length) or target_length <= 0:
        target_length = 1.0
    for local_idx, fid in enumerate(entry.feature_ids):
        direction = components_t[local_idx]
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length == 0.0:
            continue
        scaled = direction * (target_length / length)
        end_point = baseline + scaled
        fig.add_trace(
            go.Scatter3d(
                x=[float(baseline[0]), float(end_point[0])],
                y=[float(baseline[1]), float(end_point[1])],
                z=[float(baseline[2]), float(end_point[2])],
                mode="lines",
                line=dict(color="black", width=2),
                name=f"Feature {fid}",
                hovertemplate=f"Feature {fid}<extra></extra>",
                showlegend=False,
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="PC1", showticklabels=False, zeroline=False),
            yaxis=dict(title="PC2", showticklabels=False, zeroline=False),
            zaxis=dict(title="PC3", showticklabels=False, zeroline=False),
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


def write_dashboard(results: List[PCAResult], output_path: Path) -> None:
    figures = [_make_figure_spec(entry) for entry in results]
    figure_strings = [json.dumps(spec, separators=(",", ":")) for spec in figures]

    components_payload = [
        {
            "component_index": entry.component_index,
            "token_count": entry.token_count,
            "feature_ids": entry.feature_ids,
            "variance": [float(v) for v in entry.variance_ratio],
            "count_hist": entry.count_hist,
        }
        for entry in results
    ]

    total_pages = max(1, math.ceil(len(results) / PAGE_SIZE))

    cell_markup: List[str] = []
    for cell_idx in range(PAGE_SIZE):
        cell_markup.append(
            (
                "        <div class=\"cell\" data-cell=\"{idx}\">\n"
                "          <div class=\"cell-info\" id=\"info-{idx}\"></div>\n"
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
        "Colors: red = 1 feature active, orange = 2 features, blue = 3+ features."
    )

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>3D PCA Component Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-3.1.0.min.js\"></script>
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
    .cell-actions {{ margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }}
    .cell-actions button {{ padding: 0.3rem 0.6rem; font-size: 0.72rem; cursor: pointer; }}
    .cell-actions button:disabled {{ cursor: not-allowed; opacity: 0.45; }}
  </style>
</head>
<body>
  <h1>3D PCA of multi-feature components</h1>
  <div class=\"meta\">Showing {PAGE_SIZE} plots per page (4 × 2 grid). Total components: {len(results)}. Use the arrows or ←/→ keys to switch pages. {meta_note}</div>
  <div class=\"nav\">
    <button id=\"prev-page\">◀ Prev</button>
    <span id=\"page-indicator\"></span>
    <button id=\"next-page\">Next ▶</button>
    <button id=\"unload-page\">Unload this page</button>
  </div>
  <div class=\"grid\">
{cells_html}
  </div>
  <script>
    const COMPONENTS = {components_json_safe};
    const FIGURE_STRINGS = {figures_json_safe};
    const PAGE_SIZE = {PAGE_SIZE};
    const TOTAL_COMPONENTS = COMPONENTS.length;
    const TOTAL_PAGES = Math.max(1, Math.ceil(TOTAL_COMPONENTS / PAGE_SIZE));
    const FEATURE_BASE_URL = \"{FEATURE_BASE_URL}\";

    const assignments = new Array(PAGE_SIZE).fill(null);
    let currentPage = 0;

    function plotId(cell) {{ return 'plot-' + String(cell); }}
    function infoId(cell) {{ return 'info-' + String(cell); }}

    function cloneSpec(idx) {{
      return JSON.parse(FIGURE_STRINGS[idx]);
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
      const counts = formatCounts(metadata.count_hist || []);
      return [
        'Comp ' + metadata.component_index,
        'Tokens: ' + metadata.token_count,
        'Features (' + metadata.feature_ids.length + '): ' + features,
        'Variance: ' + variance,
        'Counts (1/2/3+): ' + counts,
      ].join('<br>');
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

    function assignCell(cell, globalIndex) {{
      assignments[cell] = globalIndex;
      const info = document.getElementById(infoId(cell));
      const buttons = document.querySelectorAll('.feature-button[data-cell="' + cell + '"]');
      purgeCell(cell);

      if (globalIndex === null || globalIndex >= TOTAL_COMPONENTS) {{
        if (info) info.innerHTML = '';
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
    }}

    function loadPlotByCell(cell) {{
      const globalIndex = assignments[cell];
      if (globalIndex === null || globalIndex === undefined) return;
      const container = document.getElementById(plotId(cell));
      if (!container || container.dataset.loaded === 'true') return;
      const spec = cloneSpec(globalIndex);
      Plotly.newPlot(container, spec.data, spec.layout, spec.config || {{}});
      container.dataset.loaded = 'true';
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

    metadata_first = result.first_token_idx
    metadata_last = result.last_token_idx if result.last_token_idx >= 0 else None
    resolved_first = args.first_token_idx if args.first_token_idx is not None else metadata_first
    resolved_last = args.last_token_idx if args.last_token_idx is not None else metadata_last

    matrices, snippet_lists = collect_component_matrices(
        run_dir,
        components,
        first_token_idx=resolved_first,
        last_token_idx=resolved_last,
        show_progress=not args.no_progress,
    )

    results = generate_pca_results(
        components,
        matrices,
        snippet_lists,
        min_activations=args.min_activations,
        show_progress=not args.no_progress,
    )

    if not results:
        print("No components had sufficient activations for PCA")
        return

    output_dir = args.output_dir or (metadata_dir / "component_pca_3d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "component_pca_3d_dashboard.html"
    write_dashboard(results, output_path)


if __name__ == "__main__":
    main()
