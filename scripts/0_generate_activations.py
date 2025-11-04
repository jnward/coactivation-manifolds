#!/usr/bin/env python
"""Command-line driver for activation logging and indexing."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from coactivation_manifolds.activation_pipeline import ActivationPipeline, PipelineConfig
from coactivation_manifolds.activation_reader import FeatureIndexBuilder, FeatureIndexConfig
from coactivation_manifolds.activation_writer import ActivationWriter
from coactivation_manifolds.default_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_LAYER,
    DEFAULT_SAE_NAME,
    DEFAULT_SAE_RELEASE,
)
from coactivation_manifolds.sae_loader import SAEHandle, load_sae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log SAE activations to Parquet shards")
    parser.add_argument("output_dir", type=Path, help="Root directory for activations and metadata")
    parser.add_argument("dataset", type=str, nargs="?", default="monology/pile-uncopyrighted", help="Dataset name or local path")
    parser.add_argument("split", type=str, nargs="?", default="train", help="Dataset split (default: train)")

    parser.add_argument("--dataset-config", default=None, help="Optional dataset config name")
    parser.add_argument("--text-field", default="text", help="Field containing raw text")
    parser.add_argument("--batch-size", type=int, default=2, help="Samples per forward batch")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer max length")
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after logging this many tokens")
    parser.add_argument("--shard-size", type=int, default=8192, help="Tokens per Parquet shard")
    parser.add_argument(
        "--stream-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the dataset instead of downloading locally (default: True)",
    )
    parser.add_argument("--device", default="cuda", help="Device for model/SAE (e.g., cuda, cuda:1, cpu)")
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use device_map='auto' to spread model across all available GPUs (recommended for multi-GPU setups)",
    )
    parser.add_argument("--layer-index", type=int, default=DEFAULT_LAYER, help="Model layer to tap for activations")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model id")
    parser.add_argument("--sae-release", default=DEFAULT_SAE_RELEASE, help="sae-lens release identifier")
    parser.add_argument("--sae-name", default=DEFAULT_SAE_NAME, help="sae-lens SAE identifier within the release")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    output_dir: Path = args.output_dir
    activations_dir = output_dir / "activations"
    metadata_dir = output_dir / "metadata"
    activations_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"

    # Multi-GPU support
    torch_dtype = torch.bfloat16 if "cuda" in args.device else torch.float32
    if args.multi_gpu:
        print(f"Loading model with device_map='auto' across {torch.cuda.device_count()} GPUs")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        # SAE goes on the primary device where hidden states will be
        sae_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipeline_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype)
        model.to(args.device)
        sae_device = args.device
        pipeline_device = args.device

    handle: SAEHandle = load_sae(
        sae_release=args.sae_release,
        sae_name=args.sae_name,
        device=sae_device,
    )

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=args.stream_dataset,
    )

    feature_count = handle.feature_count
    writer = ActivationWriter(activations_dir, num_features=feature_count, shard_size_tokens=args.shard_size)
    config = PipelineConfig(
        dataset=dataset,
        text_field=args.text_field,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layer_index=args.layer_index,
        device=pipeline_device,
        use_device_map=args.multi_gpu,
    )

    pipeline = ActivationPipeline(model, tokenizer, handle.sae, writer, config)
    pipeline.run(max_tokens=args.max_tokens, show_progress=True)

    index_path = metadata_dir / "feature_index.parquet"
    builder = FeatureIndexBuilder(
        FeatureIndexConfig(activations_dir, index_path, num_features=feature_count)
    )
    builder.build()

    print(f"Activation shards written to {activations_dir}")
    print(f"Feature index written to {index_path}")


if __name__ == "__main__":
    main()
