#!/usr/bin/env bash
set -euo pipefail

optimizer_name="${OPTIMIZER_NAME:-schedulefree}"
gpu_count="${SLURM_GPUS_ON_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"

data_path="${DATA_PATH:-$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000}"
project_name="${PROJECT_NAME:-arcagi}"
run_name="${RUN_NAME:-PDAT-TARM-sudoku-compact-v1}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints/${run_name}}"

mkdir -p "$checkpoint_path"

torchrun --nproc-per-node "${gpu_count}" --master_port="${MASTER_PORT}" pretrain.py \
    data_path="${data_path}" \
    arch=pdat_tarm \
    arch.task_type=sudoku \
    arch.loops=8 \
    arch.H_cycles=1 \
    arch.L_cycles=2 \
    arch.hidden_size= 256 \
    arch.num_heads=8 \
    arch.num_layers=2 \
    arch.num_factors=32 \
    arch.factor_seed_topk=12 \
    arch.entity_factor_topk=6 \
    arch.factor_entity_topk=12 \
    arch.factor_keep_ratio=1.0 \
    arch.entity_keep_ratio=1.0 \
    arch.support_tau_start=1.0 \
    arch.support_tau_end=0.1 \
    arch.conflict_tau_start=1.0 \
    arch.conflict_tau_end=0.1 \
    arch.tau_anneal_steps=20000 \
    arch.structural_factor_init=weak \
    arch.structural_prior_strength=0.25 \
    arch.entity_freeze_delta_threshold=0.003 \
    arch.entity_freeze_confidence_threshold=0.985 \
    arch.factor_freeze_delta_threshold=0.003 \
    epochs=100000 \
    eval_interval=2000 \
    project_name="${project_name}" \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0 \
    global_batch_size=100 \
    grad_accum_steps=6 \
    +run_name="${run_name}" \
    +checkpoint_path="${checkpoint_path}" \
    +ema=True \
    optimizer_name="${optimizer_name}"
