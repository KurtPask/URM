optimizer_name="schedulefree"  #"schedulefree" or "adam_atan2"
arch=marm
lc=6
hc=2
hs=512
nh=8
loop=16
nl=4
ga=1
bs=384
lr=1e-4
pz_lr=1e-2
pn=sudoku
wd=0.1
run_name="${arch}-${pn}_${ver}_loop${loop}_H${hc}_L${lc}_layers${nl}_gradaccum${ga}_hidden${hs}_head${nh}_batch${bs}_lr${lr}_${optimizer_name}"
checkpoint_path="$HOME/TropicalURM/URM/checkpoints/${run_name}" 
resume_ckpt="${checkpoint_path}/step_130200.pt"
mkdir -p $checkpoint_path
ls -lh "$checkpoint_path"/step_*.pt

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

torchrun --nproc-per-node $SLURM_GPUS_ON_NODE --master_port=${MASTER_PORT} pretrain.py \
    data_path=$HOME/TropicalURM/URM/data/sudoku-extreme-1k-aug-1000 \
    arch=$arch arch.loops=$loop arch.H_cycles=$hc arch.L_cycles=$lc arch.hidden_size=$hs arch.num_heads=$nh arch.num_layers=$nl \
    epochs=100000 \
    eval_interval=2000 \
    project_name=$pn \
    lr=$lr puzzle_emb_lr=$pz_lr weight_decay=$wd puzzle_emb_weight_decay=$wd global_batch_size=$bs grad_accum_steps=$ga \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +load_checkpoint=$resume_ckpt \
    +load_optimizer_state=False \
    +ema=True \
    optimizer_name=$optimizer_name \
    
