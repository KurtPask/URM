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

python evaluate_sudoku_checkpoints.py "$@"
