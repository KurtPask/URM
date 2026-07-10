#!/usr/bin/env bash
set -euo pipefail

echo "Notice: scripts/TAPR_arc.sh is the legacy segment-wise launcher. Use scripts/TAPR_full_arc.sh for TAPR-Full." >&2

optimizer_name="${OPTIMIZER_NAME:-schedulefree}"
gpu_count="${SLURM_GPUS_ON_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"

data_path="${DATA_PATH:-data/arc1concept-aug-1000}"
project_name="${PROJECT_NAME:-tapr-arcagi}"
run_name="${RUN_NAME:-TAPR-arc-v3-ponder-nextlat-loop18-H1-L1-layers1-hidden192-head8}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints/${run_name}}"

mkdir -p "$checkpoint_path"

torchrun --nproc-per-node "${gpu_count}" --master_port="${MASTER_PORT}" pretrain.py \
    data_path="${data_path}" \
    arch=tapr \
    arch.loops="${LOOPS:-18}" \
    arch.H_cycles="${H_CYCLES:-1}" \
    arch.L_cycles="${L_CYCLES:-1}" \
    arch.hidden_size="${HIDDEN_SIZE:-192}" \
    arch.num_heads="${NUM_HEADS:-8}" \
    arch.num_layers="${NUM_LAYERS:-1}" \
    arch.expansion="${EXPANSION:-4}" \
    arch.tropical_attention_version="${TROPICAL_ATTENTION_VERSION:-v3}" \
    arch.ponder_min_steps="${PONDER_MIN_STEPS:-2}" \
    arch.ponder_prior_lambda="${PONDER_PRIOR_LAMBDA:-0.22}" \
    arch.ponder_eval_threshold="${PONDER_EVAL_THRESHOLD:-0.55}" \
    arch.ponder_step_cost="${PONDER_STEP_COST:-0.002}" \
    arch.ponder_kl_weight="${PONDER_KL_WEIGHT:-0.01}" \
    arch.halt_correctness_weight="${HALT_CORRECTNESS_WEIGHT:-0.03}" \
    arch.nextlat_weight="${NEXTLAT_WEIGHT:-0.05}" \
    arch.nextlat_loss_type="${NEXTLAT_LOSS_TYPE:-projective}" \
    arch.tapr_distance_mode="${TAPR_DISTANCE_MODE:-hilbert}" \
    arch.fw_bias="${FW_BIAS:-0.5}" \
    arch.chart_transition="${CHART_TRANSITION:-true}" \
    arch.chart_count="${CHART_COUNT:-4}" \
    arch.chart_transition_strength="${CHART_TRANSITION_STRENGTH:-0.15}" \
    evaluators='[{name:arc@ARC}]' \
    epochs="${EPOCHS:-50000}" \
    eval_interval="${EVAL_INTERVAL:-2000}" \
    eval_max_examples_per_set="${EVAL_MAX_EXAMPLES_PER_SET:-20000}" \
    eval_example_stride="${EVAL_EXAMPLE_STRIDE:-1}" \
    project_name="${project_name}" \
    lr="${LR:-1e-4}" \
    puzzle_emb_lr="${PUZZLE_EMB_LR:-1e-4}" \
    weight_decay="${WEIGHT_DECAY:-1.0}" \
    puzzle_emb_weight_decay="${PUZZLE_EMB_WEIGHT_DECAY:-1.0}" \
    global_batch_size="${GLOBAL_BATCH_SIZE:-128}" \
    grad_accum_steps="${GRAD_ACCUM_STEPS:-1}" \
    +run_name="${run_name}" \
    +checkpoint_path="${checkpoint_path}" \
    +ema="${EMA:-True}" \
    optimizer_name="${optimizer_name}" \
    wandb_tags='[tapr,arc,ponder,nextlat,tropical_attention]'
