import os
import subprocess
from pathlib import Path

# Configure which GPUs to use; one shard per GPU.
GPU_IDS = ["0", "1", "2", "3", "5", "6", "7"]  # edit as needed

# Path to the resampling script (relative to this file).
SCRIPT_PATH = Path(__file__).parent / "resample_entropy_tokens_sharded.py"
OUTPUT_PREFIX = "data/entropy_resamples_shard"
OUTPUT_SUFFIX = ".jsonl"


def main():
    num_shards = len(GPU_IDS)
    procs = []
    for shard_idx, gpu_id in enumerate(GPU_IDS):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        cmd = [
            "python",
            str(SCRIPT_PATH),
            "--num-shards",
            str(num_shards),
            "--shard-index",
            str(shard_idx),
        ]
        print(f"Launching shard {shard_idx}/{num_shards} on GPU {gpu_id}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for completion
    exit_codes = [p.wait() for p in procs]
    for idx, code in enumerate(exit_codes):
        if code != 0:
            raise SystemExit(f"Shard {idx} exited with code {code}")
    print("All shards completed successfully.")


if __name__ == "__main__":
    main()
