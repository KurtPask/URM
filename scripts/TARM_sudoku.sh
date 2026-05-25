optimizer_name="schedulefree"  #"schedulefree" or "adam_atan2"
lc=3
hc=1
hs=1024
nh=8
loop=48
nl=1
ga=1
bs=768 
drop=0
lr=1e-3
pn=sudoku
ver=v3
halt=False
run_name="TARM-${pn}_${ver}_loop${loop}_H${hc}_L${lc}_layers${nl}_gradaccum${ga}_hidden${hs}_head${nh}_batch${bs}_lr${lr}_${optimizer_name}"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

torchrun --nproc-per-node $SLURM_GPUS_ON_NODE --master_port=${MASTER_PORT} pretrain.py \
    data_path=$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000 \
    arch=tarm arch.loops=$loop arch.H_cycles=$hc arch.L_cycles=$lc arch.hidden_size=$hs arch.num_heads=$nh arch.num_layers=$nl arch.tropical_attention_version=$ver \
    epochs=50000 \
    eval_interval=2000 \
    project_name=$pn \
    lr=$lr puzzle_emb_lr=$lr weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=$bs grad_accum_steps=$ga \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    +arch.disable_halting=$halt \
    +arch.q_dropout=$drop +arch.k_dropout=$drop +arch.v_dropout=$drop \
    optimizer_name=$optimizer_name \
    # +load_checkpoint=$HOME/TropicalURM/URM/checkpoints/URM-sudoku/step_83325.pt \
    # +load_filter_mismatched_shapes=True \
    # +load_optimizer_state=False \
    
