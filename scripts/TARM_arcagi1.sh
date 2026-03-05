run_name="TARM-arcagi1"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

torchrun --nproc-per-node=$SLURM_GPUS_ON_NODE pretrain.py \
    data_path=data/arc1concept-aug-1000 \
    arch=tarm arch.loops=1 arch.H_cycles=1 arch.L_cycles=1 arch.num_layers=4 \
    epochs=200000 \
    eval_interval=2000 \
    puzzle_emb_lr=1e-2 \
    weight_decay=0.1 \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True
