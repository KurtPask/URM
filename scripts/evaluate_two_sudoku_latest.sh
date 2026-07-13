#!/usr/bin/env bash
set -euo pipefail

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-checkpoints}"
DATA_PATH="${DATA_PATH:-data/rebuild-sudoku-extreme-1k-aug-1000}" #sudoku-tdoku-75-1k-aug-1000 #sudoku-extreme-1k-aug-1000
OUTPUT_DIR="${OUTPUT_DIR:-eval_results/sudoku}"

mkdir -p "$OUTPUT_DIR"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
CSV_PATH="${CSV_PATH:-$OUTPUT_DIR/two_model_latest_${TS}.csv}"

export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

RUNS=(
  #"TARM-sudoku_v5_loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch768_lr1e-4_adam_atan2"
  #"urm-sudoku__loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch768_lr1e-4_adam_atan2"
  #tarm-sudoku_75_v5_loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch768_lr1e-4_adam_atan2
  #urm-sudoku_75__loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch768_lr1e-4_adam_atan2
  #tarm-sudoku_v3_loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch768_lr1e-3_schedulefree
  marm-sudoku__loop16_H2_L6_layers4_gradaccum1_hidden512_head8_batch384_lr1e-4_schedulefree
)

latest_checkpoint() {
  local run_dir="$1"

  find "$run_dir" -maxdepth 1 -type f -name 'step_*.pt' -printf '%f\n' \
    | sed -E 's/^step_([0-9]+)\.pt$/\1 step_\1.pt/' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2 \
    | xargs -I{} printf '%s/%s\n' "$run_dir" "{}"
}

echo "Writing combined CSV to: $CSV_PATH"

for run in "${RUNS[@]}"; do
  run_dir="$CHECKPOINTS_ROOT/$run"

  if [[ ! -d "$run_dir" ]]; then
    echo "ERROR: missing run directory: $run_dir" >&2
    exit 1
  fi

  ckpt="$(latest_checkpoint "$run_dir")"

  if [[ -z "${ckpt:-}" ]]; then
    echo "ERROR: no step_*.pt checkpoints found in: $run_dir" >&2
    exit 1
  fi

  echo
  echo "Evaluating latest checkpoint for: $run"
  echo "Checkpoint: $ckpt"

  bash scripts/evaluate_sudoku_checkpoints.sh \
    --checkpoint-path "$ckpt" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --csv-path "$CSV_PATH"
done

echo
echo "Done. Results CSV:"
echo "$CSV_PATH"