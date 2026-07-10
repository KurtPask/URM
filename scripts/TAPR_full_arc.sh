#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

optimizer_name="${OPTIMIZER_NAME:-schedulefree}"
python_bin="${PYTHON_BIN:-python3}"
gpu_count="${SLURM_GPUS_ON_NODE:-1}"
per_gpu_batch="${PER_GPU_BATCH:-8}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH:-128}"
global_batch_size="${GLOBAL_BATCH_SIZE:-$((gpu_count * per_gpu_batch))}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-$(((target_effective_batch + global_batch_size - 1) / global_batch_size))}"

export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

data_path="${DATA_PATH:-data/arc1concept-aug-1000}"
project_name="${PROJECT_NAME:-tapr-full-arcagi}"
loops="${LOOPS:-18}"
hidden_size="${HIDDEN_SIZE:-192}"
num_heads="${NUM_HEADS:-8}"
nextlat_enabled="${NEXTLAT_ENABLED:-true}"
solution_score_enabled="${SOLUTION_SCORE_ENABLED:-true}"
if [[ "${nextlat_enabled}" == "true" ]]; then
    nextlat_weight="${NEXTLAT_WEIGHT:-0.05}"
else
    nextlat_weight="${NEXTLAT_WEIGHT:-0.0}"
fi
if [[ "${solution_score_enabled}" == "true" ]]; then
    solution_score_weight="${SOLUTION_SCORE_WEIGHT:-0.02}"
    arc_ranking_score="solution_score_logits"
else
    solution_score_weight="${SOLUTION_SCORE_WEIGHT:-0.0}"
    arc_ranking_score="frequency"
fi
variant_name="full"
if [[ "${nextlat_enabled}" != "true" ]]; then
    variant_name="ponder-no-nextlat"
fi
if [[ "${solution_score_enabled}" != "true" ]]; then
    variant_name="${variant_name}-frequency-rank"
fi
run_name="${RUN_NAME:-TAPR-Full-arc-v3-${variant_name}-loop${loops}-hidden${hidden_size}-head${num_heads}}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints/${run_name}}"

epochs="${EPOCHS:-2500}"
eval_interval="${EVAL_INTERVAL:-250}"
if (( epochs % eval_interval != 0 )); then
    echo "EPOCHS must be divisible by EVAL_INTERVAL." >&2
    exit 2
fi

echo "TAPR-Full: GPUs=${gpu_count}, local_batch=${per_gpu_batch}, global_batch=${global_batch_size}, grad_accum=${grad_accum_steps}, effective_batch=$((global_batch_size * grad_accum_steps))"

command=(
    torchrun
    --nproc-per-node "${gpu_count}"
    --master_port="${MASTER_PORT}"
    pretrain.py
    "data_path=${data_path}"
    arch=tapr_full
    "arch.loops=${loops}"
    "arch.hidden_size=${hidden_size}"
    "arch.num_heads=${num_heads}"
    "arch.num_layers=${NUM_LAYERS:-1}"
    "arch.expansion=${EXPANSION:-4}"
    "arch.bptt_window=${BPTT_WINDOW:-0}"
    "arch.ponder_min_steps=${PONDER_MIN_STEPS:-2}"
    "arch.ponder_prior_lambda=${PONDER_PRIOR_LAMBDA:-0.12}"
    "arch.ponder_kl_weight=${PONDER_KL_WEIGHT:-0.01}"
    "arch.ponder_step_cost=${PONDER_STEP_COST:-0.0}"
    "arch.ponder_ce_weight=${PONDER_CE_WEIGHT:-0.5}"
    "arch.final_ce_weight=${FINAL_CE_WEIGHT:-1.0}"
    "arch.eval_halt_mode=${EVAL_HALT_MODE:-cdf}"
    "arch.ponder_eval_cdf_threshold=${PONDER_EVAL_CDF_THRESHOLD:-0.5}"
    "arch.nextlat_enabled=${nextlat_enabled}"
    "arch.nextlat_weight=${nextlat_weight}"
    "arch.chart_transition=${CHART_TRANSITION:-true}"
    "arch.chart_count=${CHART_COUNT:-4}"
    "arch.chart_transition_strength=${CHART_TRANSITION_STRENGTH:-0.15}"
    "arch.solution_score_enabled=${solution_score_enabled}"
    "arch.solution_score_weight=${solution_score_weight}"
    "evaluators=[{name:arc@ARC,ranking_score:${arc_ranking_score}}]"
    "epochs=${epochs}"
    "eval_interval=${eval_interval}"
    "eval_max_examples_per_set=${EVAL_MAX_EXAMPLES_PER_SET:-20000}"
    "eval_example_stride=${EVAL_EXAMPLE_STRIDE:-1}"
    'loop_deltas=[0]'
    checkpoint_every_eval=true
    "project_name=${project_name}"
    "lr=${LR:-1e-4}"
    "puzzle_emb_lr=${PUZZLE_EMB_LR:-1e-4}"
    "weight_decay=${WEIGHT_DECAY:-0.1}"
    "puzzle_emb_weight_decay=${PUZZLE_EMB_WEIGHT_DECAY:-0.1}"
    "global_batch_size=${global_batch_size}"
    "grad_accum_steps=${grad_accum_steps}"
    "+grad_clip_norm=${GRAD_CLIP_NORM:-1.0}"
    "+run_name=${run_name}"
    "+checkpoint_path=${checkpoint_path}"
    "+ema=${EMA:-true}"
    "optimizer_name=${optimizer_name}"
    'wandb_tags=[tapr_full,arc,trajectory_ponder,nextlat,tropical_attention_v3]'
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

"${python_bin}" -c 'import hydra, schedulefree, torch, tropical_gemm, wandb'
mkdir -p "$checkpoint_path"
exec "${command[@]}"
