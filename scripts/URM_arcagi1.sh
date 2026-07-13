optimizer_name="adam_atan2"  #"schedulefree" or "adam_atan2"
lc=6
hc=2
hs=512
nh=8
loop=16
nl=4
ga=1
bs=32 
drop=0
lr=1e-4
pn=arcagi1
run_name="URM-${pn}_${ver}_loop${loop}_H${hc}_L${lc}_layers${nl}_gradaccum${ga}_hidden${hs}_head${nh}_batch${bs}_lr${lr}_${optimizer_name}"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

export MASTER_PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))

torchrun --nproc-per-node ${SLURM_GPUS_ON_NODE:-1} --master_port=${MASTER_PORT} pretrain.py \
    data_path=data/arc1concept-aug-1000 \
    arch=urm arch.loops=$loop arch.H_cycles=$hc arch.L_cycles=$lc arch.hidden_size=$hs arch.num_heads=$nh arch.num_layers=$nl \
    epochs=200000 \
    eval_interval=1 \
    project_name=$pn \
    lr=$lr puzzle_emb_lr=1e-2 weight_decay=0.1 puzzle_emb_weight_decay=0.1 global_batch_size=$bs grad_accum_steps=$ga \
    +run_name=$run_name \
    +checkpoint_path=$checkpoint_path \
    +ema=True \
    +arch.q_dropout=$drop +arch.k_dropout=$drop +arch.v_dropout=$drop \
    optimizer_name=$optimizer_name \
