import unittest

import torch

from models.TAPR.tapr import TAPR, TAPRLossHead


def _config(**overrides):
    config = {
        "batch_size": 2,
        "seq_len": 4,
        "puzzle_emb_ndim": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 12,
        "num_layers": 1,
        "hidden_size": 8,
        "expansion": 2,
        "num_heads": 2,
        "pos_encodings": "none",
        "forward_dtype": "float32",
        # v1 keeps this unit test independent of the tropical-gemm extension;
        # the production config selects v3.
        "tropical_attention_version": "v1",
        "tropical_proj": False,
        "loops": 3,
        "H_cycles": 1,
        "L_cycles": 1,
        "tapr_architecture": "clean_full",
        "halt_head_type": "scalar",
        "bptt_window": 0,
        "detach_recurrent_state": False,
        "detach_ponder_state": False,
        "ponder_enabled": True,
        "disable_halting": False,
        "ponder_min_steps": 2,
        "ponder_prior_lambda": 0.2,
        "ponder_kl_weight": 0.01,
        "ponder_step_cost": 0.0,
        "halt_correctness_weight": 0.0,
        "ponder_ce_weight": 0.5,
        "final_ce_weight": 1.0,
        "eval_halt_mode": "cdf",
        "nextlat_enabled": True,
        "nextlat_weight": 0.05,
        "nextlat_loss_type": "projective",
        "nextlat_supervised_only": True,
        "nextlat_normalize_to_ce": True,
        "solution_score_enabled": True,
        "solution_score_weight": 0.02,
        "log_train_step_metrics": True,
        "chart_transition": False,
    }
    config.update(overrides)
    return config


def _batch():
    torch.manual_seed(7)
    return {
        "inputs": torch.randint(0, 12, (2, 4)),
        "labels": torch.randint(0, 12, (2, 4)),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
    }


class TAPRFullTest(unittest.TestCase):
    def test_shifted_prior_is_normalized_over_decision_steps(self):
        head = TAPRLossHead(TAPR(_config()), "stablemax_cross_entropy")
        prior = head._trajectory_prior(torch.device("cpu"))
        torch.testing.assert_close(prior, torch.tensor([0.0, 0.2, 0.8]))
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=7)

        head = TAPRLossHead(
            TAPR(_config(loops=5, readout_every=2)),
            "stablemax_cross_entropy",
        )
        prior = head._trajectory_prior(torch.device("cpu"))
        torch.testing.assert_close(prior, torch.tensor([0.0, 0.2, 0.0, 0.16, 0.64]))
        self.assertAlmostEqual(float(prior.sum()), 1.0, places=7)

    def test_clean_full_rejects_conflicting_halt_objective(self):
        with self.assertRaisesRegex(ValueError, "halt_correctness_weight"):
            TAPR(_config(halt_correctness_weight=0.03))

    def test_complete_trajectory_has_unit_mass_and_one_backward(self):
        model = TAPR(_config())
        head = TAPRLossHead(model, "stablemax_cross_entropy")
        carry, loss, metrics, outputs, done = head(
            carry=None,
            batch=_batch(),
            return_keys={"logits", "solution_score_logits"},
            run_trajectory=True,
        )

        self.assertTrue(bool(done))
        self.assertTrue(bool(carry.halted.all()))
        self.assertAlmostEqual(float(metrics["ponder_mass"]) / 2.0, 1.0, places=6)
        self.assertAlmostEqual(float(metrics["ponder_prior_mass"]) / 2.0, 1.0, places=6)
        self.assertGreaterEqual(float(metrics["ponder_kl_loss"]), -1e-6)
        self.assertIn("accuracy_step_01", metrics)
        self.assertIn("accuracy_step_03", metrics)
        self.assertIn("solution_score_logits", outputs)
        self.assertFalse(any(value.requires_grad for value in metrics.values()))

        loss.backward()
        backbone_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and "layers" in name
        )
        self.assertGreater(backbone_grad, 0.0)

    def test_nextlat_online_branch_reaches_reasoning_stack(self):
        model = TAPR(_config(solution_score_enabled=False, solution_score_weight=0.0))
        batch = _batch()
        carry = model.initial_carry(batch)
        nextlat_loss = torch.zeros(())
        for _ in range(model.config.loops):
            carry, outputs = model(
                carry,
                batch,
                force_full_trajectory=True,
            )
            nextlat_loss = nextlat_loss + outputs["nextlat_loss"].sum()

        nextlat_loss.backward()
        backbone_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and "layers" in name
        )
        self.assertGreater(backbone_grad, 0.0)

    def test_terminal_tail_has_gradient_to_earlier_halt_bias(self):
        model = TAPR(_config(nextlat_enabled=False, nextlat_weight=0.0))
        batch = _batch()
        carry = model.initial_carry(batch)
        outputs = None
        for _ in range(model.config.loops):
            carry, outputs = model(carry, batch, force_full_trajectory=True)
        assert outputs is not None

        gradient = torch.autograd.grad(
            outputs["ponder_prob"].sum(),
            model.inner.halt_head.bias,
        )[0]
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
