#!/usr/bin/env python
"""Train and evaluate a linear POS probe on Gemma-2-2b."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

from pos_probe import activations, constants, data, probe, sklearn_trainer, utils, plots  # noqa: E402

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear POS probe for Gemma-2-2b.")
    parser.add_argument("--model-name", default=constants.DEFAULT_MODEL_NAME, help="HF model identifier.")
    parser.add_argument("--layer-index", type=int, default=constants.DEFAULT_LAYER_INDEX, help="0-indexed transformer layer.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenization max length.")
    parser.add_argument("--activation-cache-dir", type=Path, default=constants.DEFAULT_ACTIVATION_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=constants.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract-batch-size", type=int, default=8, help="Batch size during activation extraction.")
    parser.add_argument("--probe-batch-size", type=int, default=1024, help="Batch size for probe training.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--trainer", choices=["pytorch", "sklearn"], default="pytorch", help="Training backend to use.")
    parser.add_argument("--sklearn-max-iter", type=int, default=200, help="Max iterations for sklearn logistic regression.")
    parser.add_argument("--sklearn-C", type=float, default=1.0, help="Inverse L2 regularization strength for sklearn logistic regression.")
    parser.add_argument("--sklearn-class-weight", choices=["none", "balanced"], default="balanced", help="Class weighting scheme for sklearn logistic regression.")
    parser.add_argument("--disabled-tags", type=str, default="", help="Comma-separated POS tags to drop from training/evaluation (e.g., INTJ).")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--num-proc", type=int, default=4, help="Tokenizer multiprocessing workers.")
    parser.add_argument("--num-workers", type=int, default=2, help="PyTorch dataloader workers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-dtype", choices=list(DTYPE_MAP.keys()), default="float16")
    parser.add_argument("--activation-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--recompute-activations", action="store_true")
    parser.add_argument(
        "--evaluation-normalization",
        choices=["softmax", "relu"],
        default="softmax",
        help="Probability normalization to emphasize when reporting metrics.",
    )
    parser.add_argument(
        "--classification-mode",
        choices=["multiclass", "ovr"],
        default="multiclass",
        help="Train a single multinomial probe or one-vs-rest binary probes.",
    )
    return parser.parse_args()


def normalize_confusion_matrix(cm: np.ndarray, mode: str | None) -> np.ndarray:
    return plots.normalize_confusion_matrix(cm, mode)


def plot_confusion_matrix(*args, **kwargs) -> None:
    plots.plot_confusion_matrix(*args, **kwargs)


def _build_label_mapping(disabled_tags: set[str]) -> tuple[list[str], np.ndarray | None]:
    active = [tag for tag in constants.POS_TAGS if tag not in disabled_tags]
    if not active:
        raise ValueError("All tags were disabled; cannot train a probe.")
    if not disabled_tags:
        return active, None
    mapping = np.full(len(constants.POS_TAGS), -1, dtype=np.int64)
    for new_idx, tag in enumerate(active):
        mapping[constants.TAG_TO_ID[tag]] = new_idx
    return active, mapping


def _filter_features_labels(
    feats: torch.Tensor,
    labels: torch.Tensor,
    mapping: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels_np = labels.numpy()
    mapped = mapping[labels_np]
    keep = mapped >= 0
    keep_tensor = torch.from_numpy(keep)
    feats = feats[keep_tensor]
    mapped_labels = torch.from_numpy(mapped[keep]).long()
    return feats, mapped_labels


def build_feature_loader(
    path: Path,
    batch_size: int,
    shuffle: bool,
    mapping: np.ndarray | None = None,
) -> DataLoader:
    feats, labels = activations.load_activation_cache(path)
    if mapping is not None:
        feats, labels = _filter_features_labels(feats, labels, mapping)
    else:
        labels = labels.long()
    dataset = TensorDataset(feats, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return loader


def load_filtered_numpy(path: Path, mapping: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = activations.load_activation_cache(path)
    feats_np = feats.numpy().astype(np.float32)
    labels_np = labels.numpy()
    if mapping is not None:
        mapped = mapping[labels_np]
        keep = mapped >= 0
        feats_np = feats_np[keep]
        labels_np = mapped[keep]
    return feats_np, labels_np


def main() -> None:
    args = parse_args()
    if args.classification_mode == "ovr" and args.trainer != "sklearn":
        raise ValueError("One-vs-rest mode currently requires --trainer sklearn.")
    utils.set_seed(args.seed)
    disabled_tags = {
        tag.strip().upper()
        for tag in args.disabled_tags.split(",")
        if tag.strip()
    }
    unknown_tags = disabled_tags.difference(constants.POS_TAGS)
    if unknown_tags:
        raise ValueError(f"Unknown tags in --disabled-tags: {', '.join(sorted(unknown_tags))}")
    active_tags, label_mapping = _build_label_mapping(disabled_tags)

    tokenizer = data.load_tokenizer(args.model_name)
    raw_dataset, tag_names = data.load_ud_dataset()
    tokenized = data.tokenize_and_align(
        raw_dataset,
        tokenizer,
        tag_names=tag_names,
        max_length=args.max_length,
        num_proc=args.num_proc,
    )
    collator = data.POSDataCollator(tokenizer)

    splits_needed = {
        args.train_split,
        args.val_split,
        args.test_split,
    }

    dataloaders = {
        split: DataLoader(
            tokenized[split],
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        for split in splits_needed
    }

    cache_files = {
        split: activations.cache_path(
            args.activation_cache_dir, args.model_name, args.layer_index, split
        )
        for split in splits_needed
    }

    splits_to_compute = [
        split
        for split, path in cache_files.items()
        if args.recompute_activations or not path.exists()
    ]

    model = None
    model_hidden_size = None
    model_dtype = DTYPE_MAP[args.model_dtype]

    if splits_to_compute:
        print(f"Extracting activations for splits: {', '.join(splits_to_compute)}")
        model = activations.load_model(
            args.model_name,
            device=args.device,
            dtype=model_dtype,
        )
        model_hidden_size = model.config.hidden_size
        for split in splits_to_compute:
            feats, labels = activations.extract_activations(
                model,
                dataloaders[split],
                layer_index=args.layer_index,
                device=args.device,
                dtype=DTYPE_MAP[args.activation_dtype],
                desc=f"{split} activations",
            )
            activations.save_activation_cache(cache_files[split], feats, labels)
    else:
        print("All activation caches already present.")

    if model_hidden_size is None:
        # Peek at one cache to infer dimensionality.
        sample_feats, _ = activations.load_activation_cache(cache_files[args.train_split])
        model_hidden_size = sample_feats.shape[1]

    # We can drop the heavy model before probe training.
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_loader_feats = build_feature_loader(
        cache_files[args.train_split], args.probe_batch_size, shuffle=True, mapping=label_mapping
    )
    val_loader_feats = build_feature_loader(
        cache_files[args.val_split], args.probe_batch_size, shuffle=False, mapping=label_mapping
    )
    test_loader_feats = build_feature_loader(
        cache_files[args.test_split], args.probe_batch_size, shuffle=False, mapping=label_mapping
    )

    probe_model = probe.LinearProbe(model_hidden_size, num_labels=len(active_tags))

    if args.trainer == "pytorch":
        history = probe.train_probe(
            probe_model,
            train_loader_feats,
            val_loader_feats,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
        )
    else:
        train_feats_np, train_labels_np = load_filtered_numpy(
            cache_files[args.train_split], label_mapping
        )
        class_weight = None if args.sklearn_class_weight == "none" else "balanced"

        if args.classification_mode == "multiclass":
            clf = sklearn_trainer.train_logistic_regression(
                train_feats_np,
                train_labels_np,
                C=args.sklearn_C,
                max_iter=args.sklearn_max_iter,
                random_state=args.seed,
                class_weight=class_weight,
            )
            probe_model = sklearn_trainer.to_linear_probe(
                clf,
                hidden_size=model_hidden_size,
            ).to(args.device)
        else:
            ovr_class_weight = class_weight if class_weight is not None else "balanced"
            probe_model = sklearn_trainer.train_one_vs_rest_probe(
                train_feats_np,
                train_labels_np,
                C=args.sklearn_C,
                max_iter=args.sklearn_max_iter,
                random_state=args.seed,
                class_weight=ovr_class_weight,
                hidden_size=model_hidden_size,
            ).to(args.device)
        train_loss = probe.average_loss(
            probe_model,
            train_loader_feats,
            device=args.device,
        )
        val_loss = probe.average_loss(
            probe_model,
            val_loader_feats,
            device=args.device,
        )
        val_softmax = probe.evaluate(
            probe_model,
            val_loader_feats,
            device=args.device,
            normalization="softmax",
        )
        val_relu = probe.evaluate(
            probe_model,
            val_loader_feats,
            device=args.device,
            normalization="relu",
        )
        history = [
            probe.TrainStats(
                epoch=1,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc_softmax=val_softmax,
                val_acc_relu=val_relu,
            )
        ]

    test_loss = probe.average_loss(
        probe_model,
        test_loader_feats,
        device=args.device,
    )
    test_acc_softmax = probe.evaluate(
        probe_model,
        test_loader_feats,
        device=args.device,
        normalization="softmax",
    )
    test_recalls_softmax = probe.per_tag_recall(
        probe_model,
        test_loader_feats,
        device=args.device,
        normalization="softmax",
        tag_names=active_tags,
    )
    test_recalls_relu = probe.per_tag_recall(
        probe_model,
        test_loader_feats,
        device=args.device,
        normalization="relu",
        tag_names=active_tags,
    )
    test_confusion_softmax = probe.confusion_matrix_counts(
        probe_model,
        test_loader_feats,
        device=args.device,
        normalization="softmax",
    )
    test_acc_relu = probe.evaluate(
        probe_model,
        test_loader_feats,
        device=args.device,
        normalization="relu",
    )

    output_dir = args.output_dir / args.model_name.replace("/", "-") / f"layer_{args.layer_index}"
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "probe.pt"
    metrics_path = output_dir / "metrics.json"
    confusion_path = output_dir / "confusion_matrix_softmax.png"
    confusion_norm_path = output_dir / "confusion_matrix_softmax_normalized.png"

    config = constants.ProbeConfig(
        model_name=args.model_name,
        layer_index=args.layer_index,
        classification_mode=args.classification_mode,
        normalization=args.evaluation_normalization,
        tag_set=tuple(active_tags),
        disabled_tags=tuple(sorted(disabled_tags)),
        max_length=args.max_length,
        activation_cache_dir=args.activation_cache_dir,
    )

    probe.save_probe(
        probe_model,
        weights_path,
        metadata={
            "hidden_size": model_hidden_size,
            "num_labels": len(active_tags),
            "config": config.to_dict(),
            "trainer": args.trainer,
            "sklearn_params": {
                "C": args.sklearn_C,
                "max_iter": args.sklearn_max_iter,
                "class_weight": args.sklearn_class_weight,
            }
            if args.trainer == "sklearn"
            else None,
        },
    )

    plot_confusion_matrix(
        test_confusion_softmax,
        active_tags,
        confusion_path,
        title="Confusion Matrix (Softmax predictions)",
    )
    plot_confusion_matrix(
        test_confusion_softmax,
        active_tags,
        confusion_norm_path,
        title="Confusion Matrix (Softmax, normalized by true tag)",
        normalize="true",
    )

    metrics = {
        "train_examples": len(train_loader_feats.dataset),
        "val_examples": len(val_loader_feats.dataset),
        "test_examples": len(test_loader_feats.dataset),
        "test_loss": test_loss,
        "test_accuracy_softmax": test_acc_softmax,
        "test_accuracy_relu": test_acc_relu,
        "history": [vars(stat) for stat in history],
        "per_tag_recall_softmax": test_recalls_softmax,
        "per_tag_recall_relu": test_recalls_relu,
        "confusion_matrix_softmax": test_confusion_softmax.tolist(),
        "confusion_matrix_softmax_normalized_true": normalize_confusion_matrix(
            test_confusion_softmax, "true"
        ).tolist(),
        "trainer": args.trainer,
        "active_tags": active_tags,
        "disabled_tags": sorted(disabled_tags),
        "classification_mode": args.classification_mode,
        "sklearn_params": {
            "C": args.sklearn_C,
            "max_iter": args.sklearn_max_iter,
            "class_weight": args.sklearn_class_weight,
        }
        if args.trainer == "sklearn"
        else None,
    }
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved probe to {weights_path}")
    print(f"Test loss (cross-entropy): {test_loss:.4f}")
    print(f"Test accuracy (softmax): {test_acc_softmax:.4f}")
    print(f"Test accuracy (ReLU-norm): {test_acc_relu:.4f}")

    def format_recalls(recalls: dict[str, float | None], tags: list[str]) -> str:
        return ", ".join(
            f"{tag}:{recalls.get(tag):.2f}"
            if recalls.get(tag) is not None
            else f"{tag}:n/a"
            for tag in tags
        )

    print("Per-tag recall (softmax):")
    print(format_recalls(test_recalls_softmax, active_tags))
    print("Per-tag recall (ReLU-norm):")
    print(format_recalls(test_recalls_relu, active_tags))
    print(f"Stored confusion matrix (counts) at {confusion_path}")
    print(f"Stored normalized confusion matrix at {confusion_norm_path}")


if __name__ == "__main__":
    main()
