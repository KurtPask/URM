# Tropical Attention Pondering Reasoner

TAPR is a TARM-derived recurrent reasoning model with three additions:

- Tropical Attention remains the relational update operator.
- NextLat trains the recurrent state to predict its next latent state.
- PonderNet trains a halting distribution over recurrent steps and enables early exit at evaluation time.

The model class is `TAPR.tapr@TAPR`; the loss head is `TAPR.tapr@TAPRLossHead`.
Use `arch=tapr_full` for the trajectory-trained model. The older `arch=tapr`
configuration remains available for legacy experiments and is not TAPR-Full.

## Recurrent Step

For each recurrent step, TAPR applies:

```text
z_source = z_t + input_embedding
z_hat_next = NextLat(z_source)
z_{t+1} = TropicalAttentionBlock(z_t, input) + optional tropical chart transition
logits_t = Decoder(z_{t+1})
halt_t = HaltHead(z_{t+1}, nextlat_error, state_delta, logit_delta, step_fraction)
```

For `tapr_architecture=clean_full`, the loss head runs every recurrent step on
one batch, constructs the complete halting distribution, and performs one
backward/optimizer update for that trajectory. Evaluation reports both the
forced-final endpoint and the configured online stopping policy.

## Loss

`TAPRLossHead` optimizes:

```text
final_ce_weight * final-step task loss
+ ponder_ce_weight * expected task loss under p_n
+ nextlat_weight * CE-normalized latent prediction loss
+ ponder_kl_weight * KL(p || shifted truncated geometric prior)
+ optional task consistency and solution-score losses
```

TAPR-Full rejects `halt_correctness_weight != 0`. PonderNet's halt probability
is a compute-allocation variable, not a correctness probability. ARC candidate
ranking instead uses the separately trained `solution_score_logits` output.

The NextLat projective loss can use:

- `tapr_distance_mode=hilbert`
- `tapr_distance_mode=fw_generalized`
- `tapr_distance_mode=asymmetric_forward`
- `tapr_distance_mode=asymmetric_reverse`

`fw_generalized` implements the normalized one-infinity tropical gauge from
`Extending_the_Tropical_Fermat_Weber_Problem.pdf`.

## Metrics

The loss head emits:

- `accuracy`
- `exact_accuracy`
- `adaptive_accuracy`
- `adaptive_exact_accuracy`
- `accuracy_per_compute`
- `steps`
- `full_steps`
- `ponder_expected_steps`
- `ponder_mass`
- `ponder_mass_error`
- per-step `accuracy_step_N`, `exact_accuracy_step_N`, `halt_lambda_step_N`,
  `ponder_prob_step_N`, `ponder_mass_step_N`, and `nextlat_step_N`
- `over_halt_rate`
- `under_halt_rate`
- `halt_brier`
- `solution_score_brier`
- `nextlat_loss`
- `state_delta`
- `logit_delta`
- `chart_margin`
- `chart_gate`

Per-step metrics are emitted during evaluation by default; logging them for every
training update can be enabled with `log_train_step_metrics=true`.

## ARC launcher

Run the explicit production configuration with:

```bash
scripts/TAPR_full_arc.sh
```

The launcher derives a memory-conscious per-rank batch size, uses gradient
accumulation to target an effective batch of 128, clips full-trajectory
gradients, evaluates only the configured 18-step depth, and ranks ARC candidates
with `solution_score_logits` rather than the halt head.
