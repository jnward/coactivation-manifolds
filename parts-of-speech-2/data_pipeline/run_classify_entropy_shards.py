import argparse
import os
import subprocess
from pathlib import Path

CLASSIFIER = Path(__file__).with_name("classify_entropy_resamples.py")

GPU_SHARD_MAP = {
    "0": [0, 1, 2, 3],
    "1": [4, 5, 6, 7],
    "2": [8, 9, 10, 11],
    "3": [12, 13, 14, 15],
    "5": [16, 17, 18, 19],
    "6": [20, 21, 22, 23],
    "7": [24, 25, 26, 27],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run per-shard classification")
    parser.add_argument("--shard-prefix", default="entropy_resamples_subshard")
    parser.add_argument("--mapping-json", default=None, help="JSON dict of gpu->shard list")
    parser.add_argument("--output-suffix", default="_with_spacy_pos.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    mapping = GPU_SHARD_MAP
    if args.mapping_json:
        import json

        mapping = json.loads(args.mapping_json)

    procs = []
    for gpu, shards in mapping.items():
        for shard in shards:
            input_path = Path(f"{args.shard_prefix}{shard}.jsonl")
            output_path = input_path.with_name(input_path.stem + args.output_suffix)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd = [
                "python",
                str(CLASSIFIER),
                f"--input={input_path}",
                f"--output={output_path}",
            ]
            procs.append(subprocess.Popen(cmd, env=env))

    for proc in procs:
        proc.wait()


if __name__ == "__main__":
    main()
