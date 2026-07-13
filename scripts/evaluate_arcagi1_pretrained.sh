DISABLE_COMPILE=1 torchrun --nproc-per-node=$SLURM_GPUS_ON_NODE evaluate_trained_model.py \
  --checkpoint-path checkpoints/TARM-arcagi1_v4_loop48_H1_L3_layers1_gradaccum1_hidden1024_head8_batch192_lr0.001_schedulefree/step_164644.pt \
  --data-path data/arc1concept-aug-1000 \
  --output-dir eval_results/arcagi1 \
  --batch-size 128 \
  --evaluator arc@ARC \
  --eval-log-every-n-batches 1