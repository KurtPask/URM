#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <urm_checkpoint.pt> <tarm_checkpoint.pt> <data_path> [extra args...]"
  echo "Example:"
  echo "  $0 checkpoints/URM/step_10000.pt checkpoints/TARM/step_10000.pt data/arc1concept-aug-1000 --num-batches 50"
  exit 1
fi

URM_CKPT="$1"
TARM_CKPT="$2"
DATA_PATH="$3"
shift 3

python analyze_q_head_logits.py \
  --urm-checkpoint "${URM_CKPT}" \
  --tarm-checkpoint "${TARM_CKPT}" \
  --data-path "${DATA_PATH}" \
  "$@"
