#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/evaluate_sudoku_pretrained.sh /path/to/checkpoint.pt [dataset_dir] [output_dir]

CHECKPOINT_PATH="${1:?checkpoint path is required}"
DATA_PATH="${2:-data/sudoku-extreme-1k-aug-1000}"
OUTPUT_DIR="${3:-eval_results/sudoku}"
BATCH_SIZE="${BATCH_SIZE:-512}"

python evaluate_pretrained_sudoku.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --data-path "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}"
