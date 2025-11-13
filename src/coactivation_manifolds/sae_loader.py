"""Minimal loader for GemmaScope SAEs via sae-lens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .default_config import DEFAULT_SAE_RELEASE, DEFAULT_SAE_NAME


@dataclass(frozen=True)
class SAEHandle:
    sae: Any
    feature_count: int


def load_sae(
    *,
    sae_release: str = DEFAULT_SAE_RELEASE,
    sae_name: str = DEFAULT_SAE_NAME,
    device: str = "cuda",
) -> SAEHandle:
    """Fetch an SAE with sae-lens and expose its latent count."""

    try:
        from sae_lens import SAE  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ImportError(
            "sae-lens is required to load SAEs. Install with `pip install sae-lens`."
        ) from exc

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_name, device=device)
    feature_count = infer_feature_count(sae)
    return SAEHandle(sae=sae, feature_count=feature_count)


def infer_feature_count(sae: Any) -> int:
    """Return the SAE latent width (kept simple for research use)."""

    for attr in ("d_sae", "n_features"):
        value = getattr(sae, attr, None)
        if isinstance(value, int):
            return value

    cfg = getattr(sae, "cfg", None)
    if cfg is not None:
        for attr in ("d_sae", "d_hidden", "n_features"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int):
                return value

    raise ValueError("Unable to infer feature count from SAE; please check the release/name")
