# Tropical Attention Pondering Reasoner

TAPR is a TARM-derived recurrent reasoning model with three additions:

- Tropical Attention remains the relational update operator.
- NextLat trains the recurrent state to predict its next latent state.
- PonderNet trains a halting distribution over recurrent steps and enables early exit at evaluation time.

The model class is `TAPR.tapr@TAPR`; the loss head is `TAPR.tapr@TAPRLossHead`.

## Recurrent Step

For each recurrent step, TAPR applies:

```text
z_source = z_t + input_embedding
z_hat_next = NextLat(z_source)
z_{t+1} = TropicalAttentionBlock(z_t, input) + optional tropical chart transition
logits_t = Decoder(z_{t+1})
halt_t = HaltHead(z_{t+1}, nextlat_error, state_delta, logit_delta, step_fraction)
```

Training runs the full halting distribution until `loops`; evaluation can stop early.

## Loss

`TAPRLossHead` optimizes:

```text
pondered task loss
+ nextlat_weight * latent prediction loss
+ ponder_kl_weight * KL(halting distribution || geometric prior)
+ ponder_step_cost * expected steps
+ halt_correctness_weight * halt calibration loss
```

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
- `accuracy_per_step`
- `steps`
- `ponder_expected_steps`
- `ponder_prob`
- `ponder_mass`
- `ponder_alive`
- `halt_prob`
- `over_halt_rate`
- `under_halt_rate`
- `halt_brier`
- `nextlat_loss`
- `state_delta`
- `logit_delta`
- `chart_margin`
- `chart_gate`

These metrics are logged by the existing `pretrain.py` loop.
