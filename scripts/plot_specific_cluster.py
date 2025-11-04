#!/usr/bin/env python
"""Visualize a manually-specified cluster of SAE features using transformer hidden states."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from dotenv import load_dotenv

# Import reusable functions from visualize_transformer_pca
from visualize_transformer_pca import (
    collect_token_metadata,
    collect_transformer_states,
    generate_pca_results,
    generate_pc_triplets,
    _make_figure_spec,
)

from coactivation_manifolds.component_graph import (
    ComponentGraphConfig,
    _resolve_decoder_vectors,
)
from coactivation_manifolds.default_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER,
    NEURONPEDIA_BASE_URL,
)

FEATURE_BASE_URL = NEURONPEDIA_BASE_URL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D SVD visualization for a manually-specified list of features"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Activation run directory containing activations/ and metadata/",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="+",
        required=True,
        help="Space-separated list of feature IDs to visualize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML file path (default: feature_cluster_{ids}.html in run_dir)",
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
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to collect for visualization (default: 10000)",
    )
    parser.add_argument(
        "--first-token-idx",
        type=int,
        default=0,
        help="Inclusive token position to start reading activations (default: 0)",
    )
    parser.add_argument(
        "--last-token-idx",
        type=int,
        default=None,
        help="Exclusive token position to stop reading activations (default: None)",
    )
    parser.add_argument(
        "--min-activations",
        type=int,
        default=16,
        help="Minimum activation rows required to run SVD (default: 16)",
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
    parser.add_argument(
        "--highlight-substring",
        default=None,
        help="If provided, highlight tokens whose snippets contain this substring (case-insensitive)",
    )
    return parser.parse_args()


def write_single_plot_html(result, output_path: Path, feature_ids: List[int], highlight_substring: str | None = None) -> None:
    """Generate HTML with a single interactive 3D SVD plot."""
    triplets = generate_pc_triplets(result.n_pcs)

    # Generate figure specs for all PC triplets
    figure_specs = []
    for pc_x, pc_y, pc_z in triplets:
        spec = _make_figure_spec(result, pc_x, pc_y, pc_z)
        figure_specs.append(spec)

    # Serialize figure specs
    figure_strings = [json.dumps(spec, separators=(",", ":")) for spec in figure_specs]
    figure_strings_safe = [s.replace("</", "<\\/") for s in figure_strings]

    # Feature info
    features_str = ", ".join(str(fid) for fid in feature_ids)
    variance_str = ", ".join(f"{v:.3f}" for v in result.variance_ratio[:min(8, len(result.variance_ratio))])
    highlight_note = ""
    if highlight_substring:
        highlight_count = sum(1 for flag in result.highlight_mask if flag)
        total_points = len(result.highlight_mask)
        highlight_note = (
            f'    <div class="info-row">\n'
            f'      <span class="info-label">Highlight:</span> '
            f'tokens containing "{highlight_substring}" '
            f'({highlight_count}/{total_points})\n'
            f'    </div>\n'
        )

    feature_buttons_html = "\n".join([
        f'        <a href="{FEATURE_BASE_URL}{fid}" target="_blank" class="feature-link">Feature {fid}</a>'
        for fid in feature_ids
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Feature Cluster: {features_str}</title>
  <script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      padding: 2rem;
      background: #f6f6f6;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    h1 {{
      margin: 0 0 1rem 0;
      font-size: 1.6rem;
      text-align: center;
    }}
    .info-panel {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      max-width: 900px;
      width: 100%;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .info-row {{
      margin: 0.5rem 0;
      font-size: 0.9rem;
      color: #333;
    }}
    .info-label {{
      font-weight: 600;
      color: #000;
    }}
    .pc-triplet-nav {{
      display: flex;
      align-items: center;
      gap: 0.6rem;
      justify-content: center;
      margin: 1rem 0;
      font-size: 0.9rem;
    }}
    .pc-triplet-nav button {{
      padding: 0.4rem 0.8rem;
      font-size: 0.9rem;
      cursor: pointer;
      border: 1px solid #ccc;
      background: #fff;
      border-radius: 4px;
    }}
    .pc-triplet-nav button:hover:not(:disabled) {{
      background: #f0f0f0;
    }}
    .pc-triplet-nav button:disabled {{
      cursor: not-allowed;
      opacity: 0.4;
    }}
    .triplet-indicator {{
      font-weight: 600;
      color: #555;
      min-width: 120px;
      text-align: center;
    }}
    .plot-container {{
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 1rem;
      width: 900px;
      height: 700px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    #plot {{
      width: 100%;
      height: 100%;
    }}
    .feature-links {{
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 0.5rem;
    }}
    .feature-link {{
      display: inline-block;
      padding: 0.3rem 0.6rem;
      font-size: 0.8rem;
      background: #4A90E2;
      color: white;
      text-decoration: none;
      border-radius: 4px;
      transition: background 0.2s;
    }}
    .feature-link:hover {{
      background: #357ABD;
    }}
  </style>
</head>
<body>
  <h1>Feature Cluster Visualization</h1>

  <div class="info-panel">
    <div class="info-row">
      <span class="info-label">Features ({len(feature_ids)}):</span> {features_str}
    </div>
    <div class="info-row">
      <span class="info-label">Tokens collected:</span> {result.token_count}
    </div>
{highlight_note}
    <div class="info-row">
      <span class="info-label">Variance explained (first {min(8, len(result.variance_ratio))} components):</span> {variance_str}
    </div>
    <div class="info-row">
      <span class="info-label">Neuronpedia links:</span>
      <div class="feature-links">
{feature_buttons_html}
      </div>
    </div>
    <div class="info-row" style="margin-top: 1rem; font-size: 0.85rem; color: #666;">
      SVD basis computed from SAE decoder directions (no centering). Origin = b_dec (SAE baseline).
      Blue points = transformer activations projected onto feature subspace.
      Red lines = decoder directions with projection magnitudes.
    </div>
  </div>

  <div class="pc-triplet-nav" id="pc-nav" style="display: {'flex' if len(triplets) > 1 else 'none'};">
    <button id="prev-triplet">◀ Prev</button>
    <span class="triplet-indicator" id="triplet-indicator"></span>
    <button id="next-triplet">Next ▶</button>
  </div>

  <div class="plot-container">
    <div id="plot"></div>
  </div>

  <script>
    const FIGURE_STRINGS = {json.dumps(figure_strings_safe, separators=(",", ":"))};
    const TRIPLETS = {json.dumps(triplets, separators=(",", ":"))};
    const NUM_TRIPLETS = TRIPLETS.length;

    let currentTripletIdx = 0;

    function loadTriplet(idx) {{
      if (idx < 0 || idx >= NUM_TRIPLETS) return;

      currentTripletIdx = idx;
      const figureString = FIGURE_STRINGS[idx];
      const spec = JSON.parse(figureString);
      const container = document.getElementById('plot');

      // Clear and reload plot
      Plotly.purge(container);
      Plotly.newPlot(container, spec.data, spec.layout, spec.config || {{}});

      // Update navigation
      updateNav();
    }}

    function updateNav() {{
      const indicator = document.getElementById('triplet-indicator');
      const prevBtn = document.getElementById('prev-triplet');
      const nextBtn = document.getElementById('next-triplet');

      if (NUM_TRIPLETS > 0) {{
        const triplet = TRIPLETS[currentTripletIdx];
        const label = 'PC' + (triplet[0]+1) + '-' + (triplet[1]+1) + '-' + (triplet[2]+1) +
                      ' (' + (currentTripletIdx+1) + '/' + NUM_TRIPLETS + ')';
        if (indicator) indicator.textContent = label;
      }}

      if (prevBtn) prevBtn.disabled = currentTripletIdx <= 0;
      if (nextBtn) nextBtn.disabled = currentTripletIdx >= NUM_TRIPLETS - 1;
    }}

    function setupControls() {{
      const prevBtn = document.getElementById('prev-triplet');
      const nextBtn = document.getElementById('next-triplet');

      if (prevBtn) {{
        prevBtn.addEventListener('click', function() {{
          if (currentTripletIdx > 0) loadTriplet(currentTripletIdx - 1);
        }});
      }}

      if (nextBtn) {{
        nextBtn.addEventListener('click', function() {{
          if (currentTripletIdx < NUM_TRIPLETS - 1) loadTriplet(currentTripletIdx + 1);
        }});
      }}

      document.addEventListener('keydown', function(event) {{
        if (event.key === 'ArrowLeft' && currentTripletIdx > 0) {{
          loadTriplet(currentTripletIdx - 1);
        }} else if (event.key === 'ArrowRight' && currentTripletIdx < NUM_TRIPLETS - 1) {{
          loadTriplet(currentTripletIdx + 1);
        }}
      }});
    }}

    function init() {{
      setupControls();
      if (NUM_TRIPLETS > 0) {{
        loadTriplet(0);
      }}
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
    print(f"\nWrote visualization to {output_path}")
    print(f"  Features: {features_str}")
    print(f"  Tokens: {result.token_count}")
    print(f"  PC triplets: {len(triplets)}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    run_dir = args.run_dir
    metadata_dir = run_dir / "metadata"
    feature_ids = sorted(args.features)

    if len(feature_ids) < 3:
        print("Error: Need at least 3 features for 3D visualization")
        return

    print(f"Visualizing features: {', '.join(str(f) for f in feature_ids)}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        features_str = "_".join(str(f) for f in feature_ids[:5])
        if len(feature_ids) > 5:
            features_str += f"_plus{len(feature_ids)-5}"
        output_path = run_dir / f"feature_cluster_{features_str}.html"

    # Load full feature counts to determine SAE size
    full_counts_path = metadata_dir / "feature_counts.parquet"
    if not full_counts_path.exists():
        raise FileNotFoundError(
            f"Missing feature_counts.parquet at {full_counts_path}. "
            "This file is needed to determine the SAE's total feature count."
        )
    full_counts_table = pq.read_table(full_counts_path)
    total_features = full_counts_table.num_rows

    # Auto-detect SAE configuration from metadata if not provided
    if args.sae_release is None or args.sae_name is None:
        from coactivation_manifolds.sae_loader import DEFAULT_SAE_RELEASE, DEFAULT_SAE_NAME
        sae_release = args.sae_release or DEFAULT_SAE_RELEASE
        sae_name = args.sae_name or DEFAULT_SAE_NAME
        print(f"Using SAE: {sae_release} / {sae_name}")
    else:
        sae_release = args.sae_release
        sae_name = args.sae_name

    # Create minimal config for decoder loading
    config = ComponentGraphConfig(
        coactivations_path=metadata_dir / "dummy.parquet",  # Not used
        feature_counts_path=metadata_dir / "dummy.parquet",  # Not used
        jaccard_threshold=0.0,  # Not used
        decoder_path=args.decoder_path,
        sae_release=sae_release,
        sae_name=sae_name,
        device="cpu",
    )

    # Load decoder directions
    print("Loading decoder directions...")
    decoder_directions = _resolve_decoder_vectors(
        feature_count=total_features,
        config=config,
        metadata_dir=metadata_dir,
    )

    # Load SAE to get b_dec
    print("Loading SAE to extract b_dec...")
    from coactivation_manifolds.sae_loader import load_sae
    sae_handle = load_sae(
        sae_release=sae_release,
        sae_name=sae_name,
        device="cpu"
    )
    b_dec = sae_handle.sae.b_dec.detach().cpu().numpy().astype(np.float32)

    # Wrap features as single "component" for reusing existing functions
    components = [feature_ids]

    # Collect token metadata
    print("Collecting token metadata...")
    highlight = args.highlight_substring.lower() if args.highlight_substring else None
    tokens_per_component, snippets_per_component, highlight_masks = collect_token_metadata(
        run_dir,
        components,
        first_token_idx=args.first_token_idx,
        last_token_idx=args.last_token_idx,
        max_tokens_per_component=args.max_tokens,
        show_progress=not args.no_progress,
        highlight_substring=highlight,
    )

    if not tokens_per_component[0]:
        print("Error: No tokens found where these features activate")
        return

    print(f"Found {len(tokens_per_component[0])} tokens")

    # Load dataset
    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
        streaming=args.stream_dataset,
    )

    # Collect transformer hidden states
    print("Collecting transformer hidden states...")
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

    # Generate SVD results
    print("Computing SVD...")
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
        print("Error: Insufficient activations for SVD")
        return

    # Write output
    write_single_plot_html(results[0], output_path, feature_ids, highlight_substring=args.highlight_substring)


if __name__ == "__main__":
    main()
