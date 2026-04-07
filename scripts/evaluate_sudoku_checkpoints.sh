#!/usr/bin/env bash
set -euo pipefail

# Unified Sudoku evaluator.
#
# Examples:
#   # Evaluate latest checkpoint from every run under checkpoints/
#   bash scripts/evaluate_sudoku_checkpoints.sh
#
#   # Evaluate one specific checkpoint
#   bash scripts/evaluate_sudoku_checkpoints.sh --checkpoint-path checkpoints/my-run/step_1000.pt
#
#   # Custom locations
#   bash scripts/evaluate_sudoku_checkpoints.sh \
#     --checkpoints-root /path/to/checkpoints \
#     --data-path data/sudoku-extreme-1k-aug-1000 \
#     --output-dir eval_results/sudoku
#
# Fast/robust defaults can be overridden from caller shell:
#   EVAL_BATCH_SIZE=256 EVAL_TEST_STRIDE=2 EVAL_MAX_EXAMPLES=50000 bash scripts/evaluate_sudoku_checkpoints.sh

export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

DEFAULT_ARGS=()
if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
  DEFAULT_ARGS+=(--batch-size "${EVAL_BATCH_SIZE}")
fi
if [[ -n "${EVAL_TEST_STRIDE:-}" ]]; then
  DEFAULT_ARGS+=(--test-example-stride "${EVAL_TEST_STRIDE}")
fi
if [[ -n "${EVAL_MAX_EXAMPLES:-}" ]]; then
  DEFAULT_ARGS+=(--max-test-examples-per-set "${EVAL_MAX_EXAMPLES}")
fi

python evaluate_sudoku_checkpoints.py "${DEFAULT_ARGS[@]}" "$@"
