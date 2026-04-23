optimizer_name="schedulefree"  # "schedulefree" or "adam_atan2"
run_name="TARM-arcagi1_v2_loop16_H2_L6_layers1_gradaccum6_hidden128_head8_skedfree"
checkpoint_path="checkpoints/${run_name}"
mkdir -p "$checkpoint_path"

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

torchrun --nproc-per-node ${SLURM_GPUS_ON_NODE:-1} --master_port=${MASTER_PORT} pretrain.py \
    data_path=data/arc1concept-aug-1000 \
    arch=tarm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.hidden_size=128 arch.num_heads=8 arch.num_layers=1 \
    epochs=50000 \
    eval_interval=2000 \
    project_name=arcagi \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=2 grad_accum_steps=6 \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    optimizer_name=$optimizer_name
