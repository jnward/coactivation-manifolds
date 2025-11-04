#!/usr/bin/env python
"""Diagnostic script to check SAE decoder norms."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from dotenv import load_dotenv

from coactivation_manifolds.default_config import DEFAULT_SAE_RELEASE, DEFAULT_SAE_NAME
from coactivation_manifolds.sae_loader import load_sae
from coactivation_manifolds.component_graph import _extract_decoder_matrix


def main():
    load_dotenv()

    print(f"Loading SAE: {DEFAULT_SAE_RELEASE} / {DEFAULT_SAE_NAME}")
    sae_handle = load_sae(
        sae_release=DEFAULT_SAE_RELEASE,
        sae_name=DEFAULT_SAE_NAME,
        device="cpu"
    )

    print("Extracting decoder matrix...")
    decoder_matrix = _extract_decoder_matrix(sae_handle.sae)

    # Ensure shape is [n_features, d_model]
    if decoder_matrix.shape[0] == sae_handle.feature_count:
        decoders = decoder_matrix
    elif decoder_matrix.shape[1] == sae_handle.feature_count:
        decoders = decoder_matrix.T
    else:
        raise ValueError("Decoder matrix shape doesn't match feature count")

    print(f"Decoder matrix shape: {decoders.shape} [n_features={decoders.shape[0]}, d_model={decoders.shape[1]}]")

    # Compute norms for all decoder vectors
    print("\nComputing decoder norms...")
    norms = np.linalg.norm(decoders, axis=1)

    # Statistics
    print("\n" + "="*60)
    print("DECODER NORM STATISTICS")
    print("="*60)
    print(f"Total features: {len(norms)}")
    print(f"Min norm:       {norms.min():.6f}")
    print(f"Max norm:       {norms.max():.6f}")
    print(f"Mean norm:      {norms.mean():.6f}")
    print(f"Std norm:       {norms.std():.6f}")
    print(f"Median norm:    {np.median(norms):.6f}")

    # Check if approximately unit norm
    close_to_one = np.abs(norms - 1.0) < 1e-5
    pct_unit = (close_to_one.sum() / len(norms)) * 100
    print(f"\nFraction within 1e-5 of unit norm: {pct_unit:.2f}%")

    # Check if all equal
    norm_range = norms.max() - norms.min()
    print(f"Range (max - min): {norm_range:.6f}")

    # Sample some random decoder norms
    print("\nSample of 10 random decoder norms:")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(norms), size=min(10, len(norms)), replace=False)
    for idx in sorted(sample_indices):
        print(f"  Feature {idx:6d}: norm = {norms[idx]:.6f}")

    # Percentiles
    print("\nNorm percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th percentile: {np.percentile(norms, p):.6f}")

    # Conclusion
    print("\n" + "="*60)
    if norm_range < 1e-4 and np.abs(norms.mean() - 1.0) < 1e-4:
        print("CONCLUSION: All decoders have unit norm (±1e-4)")
        print("            → Projection bug is NOT due to varying decoder norms")
    elif norm_range < 1e-4:
        print(f"CONCLUSION: All decoders have equal norm (~{norms.mean():.6f})")
        print("            → Decoders are normalized, but not to unit length")
    else:
        print("CONCLUSION: Decoder norms vary significantly")
        print("            → Should normalize decoders before SVD")
    print("="*60)


if __name__ == "__main__":
    main()
