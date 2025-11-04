#!/usr/bin/env python
"""Data-parallel wrapper for activation generation using multiple GPUs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "0_generate_activations.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate activations in parallel across multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 7 GPUs to process 1M tokens
  python scripts/0_generate_activations_parallel.py data/output --num-gpus 7 --max-tokens 1000000

  # Custom dataset and SAE
  python scripts/0_generate_activations_parallel.py data/output --num-gpus 4 \\
    --dataset my/dataset --sae-release gemma-scope-9b-pt-res-canonical

Note: All arguments except --num-gpus and --keep-workers are passed to the underlying
0_generate_activations.py script.
        """,
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory for activations and metadata",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of GPUs to use for parallel processing",
    )
    parser.add_argument(
        "--keep-workers",
        action="store_true",
        help="Keep worker directories after merging (for debugging)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip automatic merging (run merge_worker_outputs.py manually later)",
    )

    # Capture all other arguments to pass through
    parser.add_argument(
        "passthrough_args",
        nargs=argparse.REMAINDER,
        help="Arguments to pass to 0_generate_activations.py",
    )

    return parser.parse_args()


def build_worker_command(
    output_dir: Path,
    worker_id: int,
    passthrough_args: List[str],
    max_tokens_per_worker: int | None,
) -> List[str]:
    """Build command for a single worker."""
    worker_output = output_dir / f"worker_{worker_id}"

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        str(worker_output),
    ]

    # Add passthrough arguments, replacing --max-tokens if present
    skip_next = False
    max_tokens_replaced = False

    for i, arg in enumerate(passthrough_args):
        if skip_next:
            skip_next = False
            continue

        if arg == "--max-tokens":
            if max_tokens_per_worker is not None:
                cmd.extend(["--max-tokens", str(max_tokens_per_worker)])
                max_tokens_replaced = True
            skip_next = True
        else:
            cmd.append(arg)

    # If --max-tokens wasn't in passthrough args, add it if specified
    if not max_tokens_replaced and max_tokens_per_worker is not None:
        cmd.extend(["--max-tokens", str(max_tokens_per_worker)])

    return cmd


def extract_max_tokens(passthrough_args: List[str]) -> int | None:
    """Extract --max-tokens value from passthrough arguments."""
    for i, arg in enumerate(passthrough_args):
        if arg == "--max-tokens" and i + 1 < len(passthrough_args):
            try:
                return int(passthrough_args[i + 1])
            except ValueError:
                pass
    return None


def run_workers(output_dir: Path, num_gpus: int, passthrough_args: List[str]) -> bool:
    """Run workers in parallel, each on its own GPU."""
    # Extract and divide max_tokens if specified
    total_max_tokens = extract_max_tokens(passthrough_args)
    max_tokens_per_worker = None
    if total_max_tokens is not None:
        max_tokens_per_worker = total_max_tokens // num_gpus
        print(f"Dividing {total_max_tokens} tokens across {num_gpus} GPUs")
        print(f"Each worker will process ~{max_tokens_per_worker} tokens")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start all workers
    processes = []
    print(f"\nStarting {num_gpus} workers...")

    for gpu_id in range(num_gpus):
        cmd = build_worker_command(output_dir, gpu_id, passthrough_args, max_tokens_per_worker)

        # Set CUDA_VISIBLE_DEVICES to restrict worker to single GPU
        env = {"CUDA_VISIBLE_DEVICES": str(gpu_id), "PYTHONPATH": "src"}

        print(f"\nWorker {gpu_id} (GPU {gpu_id}):")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Output: {output_dir / f'worker_{gpu_id}'}")

        process = subprocess.Popen(
            cmd,
            env={**subprocess.os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((gpu_id, process))

    # Monitor workers
    print(f"\n{'='*60}")
    print(f"All {num_gpus} workers started. Monitoring progress...")
    print(f"{'='*60}\n")

    failed_workers = []

    # Wait for all to complete
    for gpu_id, process in processes:
        stdout, _ = process.communicate()

        if process.returncode != 0:
            print(f"\n{'='*60}")
            print(f"Worker {gpu_id} FAILED with exit code {process.returncode}")
            print(f"{'='*60}")
            print(stdout)
            failed_workers.append(gpu_id)
        else:
            print(f"Worker {gpu_id} completed successfully")

    if failed_workers:
        print(f"\n{'='*60}")
        print(f"ERROR: {len(failed_workers)} worker(s) failed: {failed_workers}")
        print(f"{'='*60}")
        return False

    print(f"\n{'='*60}")
    print(f"All {num_gpus} workers completed successfully!")
    print(f"{'='*60}\n")
    return True


def merge_workers(output_dir: Path, num_gpus: int, keep_workers: bool) -> bool:
    """Merge worker outputs into unified structure."""
    merge_script = ROOT / "scripts" / "merge_worker_outputs.py"

    cmd = [
        sys.executable,
        str(merge_script),
        str(output_dir),
        "--num-workers",
        str(num_gpus),
    ]

    if keep_workers:
        cmd.append("--keep-workers")

    print("Running merge script...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env={**subprocess.os.environ, "PYTHONPATH": "src"})

    if result.returncode != 0:
        print(f"\nERROR: Merge failed with exit code {result.returncode}")
        return False

    return True


def main() -> None:
    args = parse_args()

    if args.num_gpus < 1:
        print("ERROR: --num-gpus must be at least 1")
        sys.exit(1)

    if args.num_gpus == 1:
        print("WARNING: --num-gpus=1, consider using 0_generate_activations.py directly")

    print(f"{'='*60}")
    print(f"Data-Parallel Activation Generation")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of GPUs:   {args.num_gpus}")
    print(f"{'='*60}\n")

    # Run workers
    success = run_workers(args.output_dir, args.num_gpus, args.passthrough_args)
    if not success:
        sys.exit(1)

    # Merge results
    if not args.no_merge:
        print("\nMerging worker outputs...")
        success = merge_workers(args.output_dir, args.num_gpus, args.keep_workers)
        if not success:
            sys.exit(1)

        print("\n" + "="*60)
        print("SUCCESS: Parallel activation generation complete!")
        print("="*60)
        print(f"Merged output: {args.output_dir}/activations/")
        print(f"Metadata:      {args.output_dir}/metadata/")
        if not args.keep_workers:
            print("Worker directories have been cleaned up")
    else:
        print("\n" + "="*60)
        print("Workers completed. Run merge manually:")
        print(f"  python scripts/merge_worker_outputs.py {args.output_dir} --num-workers {args.num_gpus}")
        print("="*60)


if __name__ == "__main__":
    main()
