#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="$ROOT_DIR/src"

ENV_FILE="$ROOT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

COLLECT_SCRIPT="$ROOT_DIR/visualize_probability_simplex/collect_joint_snippets.py"
SCORE_SCRIPT="$ROOT_DIR/visualize_probability_simplex/score_definitions.py"
PLOT_SCRIPT="$ROOT_DIR/visualize_probability_simplex/plot_probability_simplex.py"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_dir> [max_examples]" >&2
  exit 1
fi

RUN_DIR="$1"
MAX_EXAMPLES="${2:-500}"
FEATURES=(36693 43412 58423)

CHOICE_A="by the common action of : jointly engaging. Example: shared the work between the two of them"
CHOICE_B="in point of comparison of. Example: little difference between the two cars"
CHOICE_C="in the time, space, or interval that separates. Example: the alley between the butcher shop and the pharmacy"

MODEL_NAME="google/gemma-3-27b-it"

SNIPPET_PATH="$RUN_DIR/metadata/probability_simplex/snippets_${FEATURES[0]}_${FEATURES[1]}_${FEATURES[2]}.parquet"
SCORED_PATH="${SNIPPET_PATH%.parquet}_scored.parquet"

# echo "[1/3] Collecting snippets -> $SNIPPET_PATH"
# python "$COLLECT_SCRIPT" \
#   "$RUN_DIR" \
#   --features "${FEATURES[@]}" \
#   --max-examples "$MAX_EXAMPLES" \
#   --output "$SNIPPET_PATH"

echo "[2/3] Scoring definitions -> $SCORED_PATH"
python "$SCORE_SCRIPT" \
  "$SNIPPET_PATH" \
  --choice "A=$CHOICE_A" \
  --choice "B=$CHOICE_B" \
  --choice "C=$CHOICE_C" \
  --batch-size 4 \
  --model-name "$MODEL_NAME"

echo "[3/3] Plotting probability simplex"
python "$PLOT_SCRIPT" \
  "$SCORED_PATH"

echo "Done. Outputs:"
echo "  Snippets: $SNIPPET_PATH"
echo "  Scored:   $SCORED_PATH"
echo "  Plot:     ${SCORED_PATH%.parquet}_pca.png"
