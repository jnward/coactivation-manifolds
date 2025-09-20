#!/usr/bin/env python
"""Compute connected components after pruning coactivation edges by Jaccard/cosine."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coactivation_manifolds.component_graph import (
    ComponentGraphConfig,
    compute_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune coactivation graph and count components")
    parser.add_argument(
        "coactivations_path",
        type=Path,
        help="Path to coactivations.parquet",
    )
    parser.add_argument(
        "feature_counts_path",
        type=Path,
        help="Path to feature_counts_trimmed.parquet",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.8,
        help="Minimum Jaccard similarity to keep an edge (default: 0.8)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=None,
        help="Maximum cosine similarity to keep an edge (optional)",
    )
    parser.add_argument(
        "--decoder-path",
        type=Path,
        default=None,
        help="Decoder matrix (.npy/.npz). When omitted, cached under metadata/decoder_directions.npy",
    )
    parser.add_argument(
        "--sae-release",
        default=None,
        help="sae-lens release identifier (defaults to 0_generate_activations default)",
    )
    parser.add_argument(
        "--sae-name",
        default=None,
        help="sae-lens SAE identifier within the release",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the SAE on when generating decoder directions (default: cpu)",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=None,
        help="Remove features whose activation rate exceeds this fraction of tokens",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000_000,
        help="Row batch size when streaming the Parquet file (default: 1e6)",
    )
    return parser.parse_args()


def _to_numpy(matrix) -> np.ndarray:
    candidate = matrix
    if hasattr(candidate, "weight"):
        candidate = candidate.weight
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    array = np.asarray(candidate)
    if array.ndim != 2:
        raise ValueError("Decoder weights must be 2D")
    return array.astype(np.float32, copy=False)


def _load_array(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        if not data.files:
            raise ValueError(f"Empty npz archive at {path}")
        array = data[data.files[0]]
        data.close()
    else:
        array = data
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Decoder weights must be 2D")
    return arr


def _extract_decoder_matrix(sae) -> np.ndarray:
    for attr in ("decoder", "W_dec", "decoder_weight"):
        candidate = getattr(sae, attr, None)
        if candidate is None:
            continue
        try:
            return _to_numpy(candidate)
        except ValueError:
            continue
    raise ValueError("Unable to locate decoder weights on SAE")


def _resolve_decoder_vectors(
    *,
    feature_count: int,
    decoder_path: Path | None,
    metadata_dir: Path,
    sae_release: str,
    sae_name: str,
    device: str,
) -> Tuple[np.ndarray, Path]:
    target_path = Path(decoder_path) if decoder_path is not None else metadata_dir / "decoder_directions.npy"
    if target_path.exists():
        vectors = _load_array(target_path)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        handle = load_sae(sae_release=sae_release, sae_name=sae_name, device=device)
        decoder = _extract_decoder_matrix(handle.sae)
        if decoder.shape[0] == feature_count:
            vectors = decoder
        elif decoder.shape[1] == feature_count:
            vectors = decoder.T
        else:
            raise ValueError("Decoder matrix does not match feature count")
        vectors = vectors.astype(np.float32, copy=False)
        np.save(target_path, vectors)
        print(f"Saved decoder directions to {target_path}")
    if vectors.shape[0] != feature_count:
        if vectors.shape[1] == feature_count:
            vectors = vectors.T
        else:
            raise ValueError("Decoder matrix does not cover all feature IDs")
    return vectors.astype(np.float32, copy=False), target_path


def main() -> None:
    args = parse_args()

    config = ComponentGraphConfig(
        coactivations_path=args.coactivations_path,
        feature_counts_path=args.feature_counts_path,
        jaccard_threshold=args.jaccard_threshold,
        cosine_threshold=args.cosine_threshold,
        decoder_path=args.decoder_path,
        sae_release=args.sae_release or ComponentGraphConfig.__dataclass_fields__["sae_release"].default,
        sae_name=args.sae_name or ComponentGraphConfig.__dataclass_fields__["sae_name"].default,
        device=args.device,
        density_threshold=args.density_threshold,
        batch_size=args.batch_size,
    )

    result = compute_components(config)

    components = [c for c in result.components if len(c) > 1]
    thresholds = [2, 3, 5, 10]
    counts_by_threshold = {t: sum(1 for comp in components if len(comp) > t) for t in thresholds}
    largest = max((len(comp) for comp in components), default=0)
    features_in_multis = sum(len(comp) for comp in components)

    print(f"Jaccard threshold: {args.jaccard_threshold}")
    if args.cosine_threshold is not None:
        print(f"Cosine threshold: {args.cosine_threshold}")
    if args.density_threshold is not None:
        print(f"Density threshold: {args.density_threshold}")
        print(f"Features removed for density: {result.removed_for_density}")
    print(f"Active features: {len(result.active_features)}")
    print(f"Edges kept: {result.edges_kept}")
    print(f"Singleton components: {result.singleton_components}")
    print(f"Components (size>1): {len(components)}")
    for t in thresholds:
        print(f"Components (size>{t}): {counts_by_threshold[t]}")
    print(f"Largest component size: {largest}")
    print(f"Features in multi-component clusters: {features_in_multis}")


if __name__ == "__main__":
    main()
