run_name="URM-sudoku-probe"
checkpoint_path="checkpoints/${run_name}"
probe_dir="analysis/activation_probe/${run_name}"
wandb_group="${WANDB_GROUP:-sudoku-urm-vs-tarm-probe}"
mkdir -p "$checkpoint_path" "$probe_dir"

if [ -n "$WANDB_API_KEY" ]; then
  wandb login "$WANDB_API_KEY" || true
fi

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))
torchrun --nproc-per-node ${SLURM_GPUS_ON_NODE:-2} --master_port=${MASTER_PORT} pretrain.py \
  data_path=$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000 \
  arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
  epochs=2500 eval_interval=100 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=100 \
  +run_name=$run_name +checkpoint_path=$checkpoint_path +ema=True \
  project_name=urm_tarm_activation_probe wandb_group=$wandb_group wandb_job_type=pretrain wandb_tags='[sudoku,probe,urm]' \
  probe.enabled=true probe.save_dir=analysis/activation_probe
