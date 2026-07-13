optimizer_name="schedulefree"  #"schedulefree" or "adam_atan2"
arch=tarm
lc=1
hc=1
hs=256
nh=16
loop=1
nl=64
ga=2
bs=384
drop=0.0
lr=1e-4
pz_lr=1e-2
pn=sudoku
wd=0.1
ver=v3
halt=False
run_name="${arch}-${pn}_${ver}_loop${loop}_H${hc}_L${lc}_layers${nl}_gradaccum${ga}_hidden${hs}_head${nh}_batch${bs}_lr${lr}_${optimizer_name}"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

torchrun --nproc-per-node $SLURM_GPUS_ON_NODE --master_port=${MASTER_PORT} pretrain.py \
    data_path=$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000 \
    arch=$arch arch.loops=$loop arch.H_cycles=$hc arch.L_cycles=$lc arch.hidden_size=$hs arch.num_heads=$nh arch.num_layers=$nl arch.tropical_attention_version=$ver \
    epochs=50000 \
    eval_interval=2000 \
    project_name=$pn \
    lr=$lr puzzle_emb_lr=$pz_lr weight_decay=$wd puzzle_emb_weight_decay=$wd global_batch_size=$bs grad_accum_steps=$ga \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    +arch.disable_halting=$halt \
    +arch.q_dropout=$drop +arch.k_dropout=$drop +arch.v_dropout=0 \
    optimizer_name=$optimizer_name \
    
