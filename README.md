# Universal Reasoning Model

[![paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.14693)

Universal transformers (UTs) have been widely used for complex reasoning tasks such as ARC-AGI and Sudoku, yet the specific sources of their performance gains remain underexplored. In this work, we systematically analyze UTs variants and show that improvements on ARC-AGI primarily arise from the recurrent inductive bias and strong nonlinear components of Transformer, rather than from elaborate architectural designs. Motivated by this finding, we propose the Universal Reasoning Model (URM), which enhances the UT with short convolution and truncated backpropagation. Our approach substantially improves reasoning performance, achieving state-of-the-art 53.8% pass@1 on ARC-AGI 1 and 16.0% pass@1 on ARC-AGI 2.

## ⚠️ For Question Regarding Sudoku Score
The reported score of 87.4% in the TRM paper is obtained using an MLP model, which we believe it is completely different from the TRM architecture in ARC-AGI task. Therefore, **_for fair comparison_**, when reproducing the results, we unified the architectures for ARC-AGI 1, ARC-AGI 2, and Sudoku to be exactly the same, which means **the architecture used to reproduce Sudoku is the same TRM architecture used to run ARC-AGI**.

Reproducing the correct TRM Sudoku score:
```bash
git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels
cd TinyRecursiveModels
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Results:

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/c0699d98-64d1-4f41-9c8f-818f924ad77b" />


## Installation
```bash
pip install -r requirements.txt
```

## Login Wandb
```bash
wandb login YOUR_API_KEY
```

## Preparing Data
```bash
# ARC-AGI-1
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m data.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Sudoku
python data/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
```

## Reproducing ARC-AGI 1 Score
```bash
bash scripts/URM_arcagi1.sh
```

## Reproducing ARC-AGI 2 Score
```bash
bash scripts/URM_arcagi2.sh
```

## Reproducing Sudoku Score
```bash
bash scripts/URM_sudoku.sh
```


## Evaluating a Pretrained Sudoku Checkpoint
```bash
# Direct Python entrypoint
python evaluate_pretrained_sudoku.py \
  --checkpoint-path checkpoints/URM-sudoku/step_83325.pt \
  --data-path data/sudoku-extreme-1k-aug-1000 \
  --output-dir eval_results/sudoku

# Or helper shell wrapper
bash scripts/evaluate_sudoku_pretrained.sh checkpoints/URM-sudoku/step_83325.pt
```

Notes:
- The Sudoku evaluator expects the dataset layout produced by `data/build_sudoku_dataset.py` (`train/`, `test/`, and `identifiers.json`).
- Evaluation always calls `model.eval()`, so dropout is disabled automatically during inference.

### Citation
```
@misc{gao2025universalreasoningmodel,
      title={Universal Reasoning Model}, 
      author={Zitian Gao and Lynx Chen and Yihao Xiao and He Xing and Ran Tao and Haoming Luo and Joey Zhou and Bryan Dai},
      year={2025},
      eprint={2512.14693},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.14693}, 
}
```


## Comparing URM vs TARM q-head logits
```bash
bash scripts/compare_q_head_logits.sh   checkpoints/URM/step_83325.pt   checkpoints/TARM/step_83325.pt   data/sudoku-extreme-1k-aug-1000   --split test --num-batches 50 --batch-size 64
```

This runs `analyze_q_head_logits.py`, prints summary statistics for `q_halt_logits`
and `q_continue_logits`, and saves an overlay histogram plot at
`analysis/q_head_logits_comparison.png` (override with `--output-plot`).

## Optimizer choice for pretraining
You can now choose the pretraining optimizer via Hydra:
```bash
python pretrain.py ... optimizer_name=adam_atan2   # default
python pretrain.py ... optimizer_name=schedulefree
```

For TARM Sudoku runs, `scripts/TARM_sudoku.sh` now defaults to `optimizer_name="schedulefree"` (override by changing that variable).

## Activation probe mode (URM vs TARM)
Probe mode captures pre-argmax/pre-threshold activation distributions during Sudoku pretraining for:
- `hidden_states_final`
- `new_carry_current_hidden`
- `output_logits` (raw lm head output)
- `q_logits_raw`
- `q_halt_logits`
- `q_continue_logits`

It is disabled by default (`probe.enabled=false`). When enabled, `pretrain.py` automatically skips `torch.compile` to keep probe instrumentation safe/reliable.

### Run URM-only probe
```bash
bash scripts/URM_sudoku_probe.sh
```

### Run TARM-only probe
```bash
bash scripts/TARM_sudoku_probe.sh
```

### Run sequential URM then TARM probe (single HPC entrypoint)
```bash
bash scripts/profile_urm_vs_tarm_sudoku.sh
```

### Local saved samples
Raw sampled tensors and summaries are saved under:
```text
analysis/activation_probe/<run_name>/step_<global_step>/
```
with:
- `summary.pt`
- `raw_samples.pt`
- `metadata.json`

### W&B logging
Scalars are logged under `probe/<tensor>/<stat>` and histograms (interval-gated) under `probe_hist/<tensor>`.
Use Hydra overrides like:
```bash
probe.enabled=true probe.interval_steps=100 probe.histogram_interval_steps=500 probe.raw_save_interval_steps=1000
project_name=urm_tarm_activation_probe wandb_group=my_group wandb_tags='[sudoku,probe]'
```

### Offline helper
```bash
python analyze_activation_probe.py \
  --urm-dir analysis/activation_probe/URM-sudoku-probe \
  --tarm-dir analysis/activation_probe/TARM-sudoku-probe \
  --step 1000 \
  --tensor hidden_states_final
```
Probe mode adds overhead and is intended for diagnostics rather than full-speed long pretraining.
