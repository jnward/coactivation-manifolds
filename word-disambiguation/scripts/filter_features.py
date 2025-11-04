#!/usr/bin/env python3
"""Filter SAE features using Neuronpedia examples and single-word heuristics."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import time

import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
WORD_DISAMBIG_DIR = SCRIPT_DIR.parent
AUTOINTERP_DIR = WORD_DISAMBIG_DIR / "automated-interpretability"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if AUTOINTERP_DIR.exists() and str(AUTOINTERP_DIR) not in sys.path:
    sys.path.insert(0, str(AUTOINTERP_DIR))

from autointerp_poc import (
    fetch_neuronpedia_data,
    format_activation_records,
    parse_sae_name_to_neuronpedia_format,
)
from neuron_explainer.activations.activation_records import ActivationRecord

DEFAULT_SAE_NAME = "google/gemma-scope-2b-pt-res/layer_12/width_65k/average_l0_72"


@dataclass
class FeatureEvaluation:
    feature_id: int
    canonical_token: Optional[str]
    valid_contexts: int
    evaluated_contexts: int
    total_contexts: int
    passed: bool
    failure_reason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-name",
        default=DEFAULT_SAE_NAME,
        help=f"SAE identifier (default: {DEFAULT_SAE_NAME})",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="*",
        default=None,
        help="Space-separated list of feature ids to evaluate",
    )
    parser.add_argument(
        "--feature-file",
        type=Path,
        default=None,
        help="Optional file containing one feature id per line",
    )
    parser.add_argument(
        "--feature-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Inclusive feature id range to evaluate",
    )
    parser.add_argument(
        "--max-nonzero",
        type=int,
        default=10,
        help="Maximum non-zero tokens allowed per context (default: 10)",
    )
    parser.add_argument(
        "--num-contexts",
        type=int,
        default=24,
        help="Number of top Neuronpedia activation records to evaluate (default: 24)",
    )
    parser.add_argument(
        "--min-valid-contexts",
        type=int,
        default=22,
        help="Minimum contexts that must satisfy heuristics (default: 22 of the top 24)",
    )
    parser.add_argument(
        "--min-total-contexts",
        type=int,
        default=45,
        help="Required Neuronpedia contexts; skip feature if fewer (default: 45)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output file for full results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-feature diagnostics to stderr",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of concurrent Neuronpedia requests (default: 8)",
    )
    return parser.parse_args()


def gather_feature_ids(args: argparse.Namespace) -> List[int]:
    ids: List[int] = []
    if args.features:
        ids.extend(args.features)
    if args.feature_file:
        with args.feature_file.open() as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                ids.append(int(stripped))
    if args.feature_range:
        start, end = args.feature_range
        ids.extend(range(start, end + 1))

    deduped = sorted(dict.fromkeys(ids))
    return deduped


def normalize_token(token: str) -> Optional[str]:
    """Strip formatting, lowercase, and ensure purely alphabetic content."""
    token = token.strip()
    if token.startswith("▁"):
        token = token.lstrip("▁")
    token = token.strip().lower()
    if not token or not token.isalpha():
        return None
    return token


def count_nonzero_tokens(record: ActivationRecord) -> int:
    return sum(1 for value in record.activations if value > 0)


def max_activation_token(record: ActivationRecord) -> Optional[str]:
    if not record.tokens or not record.activations:
        return None
    max_idx = max(range(len(record.activations)), key=lambda idx: record.activations[idx])
    return record.tokens[max_idx]


def fetch_activation_records_with_retry(
    model_id: str,
    sae_id: str,
    feature_id: int,
    *,
    max_retries: int = 7,
    base_delay: float = 60.0,
) -> List[ActivationRecord]:
    """Fetch activation records from Neuronpedia with exponential backoff on 429s."""
    attempt = 0
    while True:
        try:
            data = fetch_neuronpedia_data(model_id, sae_id, feature_id)
            return format_activation_records(data)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 429 and attempt < max_retries - 1:
                retry_after = exc.response.headers.get("Retry-After") if exc.response else None
                header_delay = float(retry_after) if retry_after else 0.0
                delay = max(base_delay * (2 ** attempt), header_delay, 1.0)
                print(
                    f"Rate limited for feature {feature_id}. "
                    f"Retrying in {delay/60:.1f} minutes..."
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise


def analyze_feature(
    feature_id: int,
    sae_name: str,
    *,
    max_nonzero: int,
    min_valid_contexts: int,
    min_total_contexts: int,
    num_contexts: int,
) -> FeatureEvaluation:
    model_id, sae_id, _ = parse_sae_name_to_neuronpedia_format(sae_name, feature_id)
    try:
        activation_records = fetch_activation_records_with_retry(model_id, sae_id, feature_id)
    except requests.HTTPError as exc:
        return FeatureEvaluation(
            feature_id=feature_id,
            canonical_token=None,
            valid_contexts=0,
            evaluated_contexts=0,
            total_contexts=0,
            passed=False,
            failure_reason=f"Neuronpedia request failed ({exc.response.status_code})",
        )

    total_contexts = len(activation_records)
    if total_contexts < min_total_contexts:
        return FeatureEvaluation(
            feature_id=feature_id,
            canonical_token=None,
            valid_contexts=0,
            evaluated_contexts=0,
            total_contexts=total_contexts,
            passed=False,
            failure_reason=f"only {total_contexts} contexts",
        )

    records_to_evaluate = activation_records[: min(num_contexts, total_contexts)]
    evaluated_contexts = len(records_to_evaluate)
    if evaluated_contexts == 0:
        return FeatureEvaluation(
            feature_id=feature_id,
            canonical_token=None,
            valid_contexts=0,
            evaluated_contexts=0,
            total_contexts=total_contexts,
            passed=False,
            failure_reason="no contexts evaluated",
        )

    canonical_token: Optional[str] = None
    valid_contexts = 0

    for record in records_to_evaluate:
        nonzero = count_nonzero_tokens(record)
        if nonzero == 0 or nonzero > max_nonzero:
            continue
        token = max_activation_token(record)
        if token is None:
            continue
        normalized = normalize_token(token)
        if not normalized:
            continue
        if canonical_token is None:
            canonical_token = normalized
        elif normalized != canonical_token:
            continue
        valid_contexts += 1

    passed = canonical_token is not None and valid_contexts >= min_valid_contexts
    failure_reason = None if passed else "insufficient matching contexts"
    return FeatureEvaluation(
        feature_id=feature_id,
        canonical_token=canonical_token,
        valid_contexts=valid_contexts,
        evaluated_contexts=evaluated_contexts,
        total_contexts=total_contexts,
        passed=passed,
        failure_reason=failure_reason,
    )


CSV_COLUMNS = [
    "feature_id",
    "canonical_token",
    "valid_contexts",
    "evaluated_contexts",
    "total_contexts",
    "passed",
    "failure_reason",
]


def load_existing_csv(path: Path) -> set[int]:
    existing: set[int] = set()
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row.get("feature_id")
            if fid is None:
                continue
            try:
                existing.add(int(fid))
            except ValueError:
                continue
    return existing


def prepare_csv_output(path: Path) -> tuple[set[int], csv.writer, any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    existing_ids = load_existing_csv(path) if file_exists else set()
    if not file_exists:
        needs_header = True
    else:
        needs_header = path.stat().st_size == 0
    if needs_header:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_COLUMNS)
    fh = path.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)
    return existing_ids, writer, fh


def append_csv_row(writer: csv.writer, evaluation: FeatureEvaluation, file_handle) -> None:
    writer.writerow(
        [
            evaluation.feature_id,
            evaluation.canonical_token or "",
            evaluation.valid_contexts,
            evaluation.evaluated_contexts,
            evaluation.total_contexts,
            1 if evaluation.passed else 0,
            evaluation.failure_reason or "",
        ]
    )
    file_handle.flush()


def print_summary(summary: dict, evaluations: Iterable[FeatureEvaluation]) -> None:
    print(summary)
    for ev in evaluations:
        if not ev.passed:
            continue
        print(
            f"Feature {ev.feature_id}: token='{ev.canonical_token}' "
            f"valid={ev.valid_contexts}/{ev.evaluated_contexts}"
        )


def main() -> None:
    args = parse_args()
    feature_ids = gather_feature_ids(args)
    if not feature_ids:
        print("No feature IDs provided. Use --features, --feature-file, or --feature-range.", file=sys.stderr)
        sys.exit(1)

    output_is_csv = bool(args.output and args.output.suffix.lower() == ".csv")
    csv_writer = None
    csv_handle = None
    skipped_existing = 0

    if output_is_csv:
        existing_ids, csv_writer, csv_handle = prepare_csv_output(args.output)
        initial_count = len(feature_ids)
        feature_ids = [fid for fid in feature_ids if fid not in existing_ids]
        skipped_existing = initial_count - len(feature_ids)
        if skipped_existing:
            print(f"Skipping {skipped_existing} feature(s) already present in {args.output}")
        if not feature_ids:
            print("All requested features are already present in the CSV. Nothing to do.")
            if csv_handle:
                csv_handle.close()
            return

    evaluations: List[FeatureEvaluation] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(
                analyze_feature,
                fid,
                args.sae_name,
                max_nonzero=args.max_nonzero,
                min_valid_contexts=args.min_valid_contexts,
                min_total_contexts=args.min_total_contexts,
                num_contexts=args.num_contexts,
            )
            for fid in feature_ids
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Features"):
            evaluation = future.result()
            evaluations.append(evaluation)
            if output_is_csv and csv_writer and csv_handle:
                append_csv_row(csv_writer, evaluation, csv_handle)
            if args.verbose:
                status = "PASS" if evaluation.passed else "FAIL"
                print(
                    f"[{status}] Feature {evaluation.feature_id}: "
                    f"token={evaluation.canonical_token} "
                    f"valid={evaluation.valid_contexts}/{evaluation.evaluated_contexts} "
                    f"reason={evaluation.failure_reason}",
                    file=sys.stderr,
                )

    evaluations.sort(key=lambda ev: ev.feature_id)
    if csv_handle:
        csv_handle.close()

    passed = sum(1 for ev in evaluations if ev.passed)
    summary = {
        "total_checked": len(evaluations),
        "passed": passed,
        "failed": len(evaluations) - passed,
    }
    if output_is_csv:
        summary["skipped_existing"] = skipped_existing

    if args.output:
        print(f"Appended {len(evaluations)} result(s) to {args.output}")
    else:
        print_summary(summary, evaluations)


if __name__ == "__main__":
    main()
