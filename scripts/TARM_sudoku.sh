optimizer_name="adam_atan2"  #"schedulefree"
run_name="TARM-sudoku_v2"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))
torchrun --nproc-per-node $SLURM_GPUS_ON_NODE --master_port=${MASTER_PORT} pretrain.py \
    data_path=$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000 \
    arch=tarm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.hidden_size=512 \
    epochs=50000 \
    eval_interval=2000 \
    project_name=arcagi \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=100 \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    optimizer_name=$optimizer_name \
    # +load_checkpoint=$HOME/TropicalURM/URM/checkpoints/URM-sudoku/step_83325.pt \
    # +load_filter_mismatched_shapes=True \
    # +load_optimizer_state=False \
    

