from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Set, Tuple
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    ConvSwiGLU,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    TropicalAttention,
    TropicalAttentionV3,
    TropicalAttentionV4,
)
from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy, softmax_cross_entropy
from models.sparse_embedding import CastedSparseEmbedding


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1.0 - 1e-5)
    return math.log(p / (1.0 - p))


def generalized_tropical_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: str = "hilbert",
    bias: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Projective tropical distance for latent-state prediction.

    `fw_generalized` implements the normalized one-infinity tropical gauge from
    the Fermat-Weber paper:
        gamma_lambda(d) = alpha max(d) - beta min(d) + delta sum(d),
    with alpha=(1-lambda)n, beta=lambda n, delta=2lambda-1.
    """
    diff = (x - y).to(torch.float32)
    max_diff = diff.amax(dim=-1)
    min_diff = diff.amin(dim=-1)

    if mode in {"hilbert", "tropical", "symmetric"}:
        return max_diff - min_diff

    n = diff.shape[-1]
    centered_sum = diff.sum(dim=-1)

    if mode == "asymmetric_forward":
        bias = 1.0
    elif mode == "asymmetric_reverse":
        bias = 0.0
    elif mode != "fw_generalized":
        raise ValueError(
            "tapr_distance_mode must be one of: hilbert, fw_generalized, "
            "asymmetric_forward, asymmetric_reverse."
        )

    bias_t = torch.as_tensor(bias, dtype=diff.dtype, device=diff.device).clamp(eps, 1.0 - eps)
    alpha = (1.0 - bias_t) * n
    beta = bias_t * n
    delta = 2.0 * bias_t - 1.0
    return (alpha * max_diff - beta * min_diff + delta * centered_sum) / max(n, 1)


@dataclass
class TAPRCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    ponder_alive: Optional[torch.Tensor] = None
    ponder_mass: Optional[torch.Tensor] = None
    ponder_expected_steps: Optional[torch.Tensor] = None
    prev_logits: Optional[torch.Tensor] = None


class TAPRConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    tropical_proj: bool = True
    tropical_qkv_proj: bool = False
    tropical_norm: str = "none"
    tropical_attention_version: str = "v3"
    attn_dropout: float = 0.0
    q_dropout: float = 0.0
    k_dropout: float = 0.0
    v_dropout: float = 0.0
    mlp_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int = 16
    L_cycles: int = 1
    H_cycles: int = 1
    forward_dtype: str = "bfloat16"

    # PonderNet-style adaptive compute.
    ponder_enabled: bool = True
    ponder_min_steps: int = 2
    ponder_eval_threshold: float = 0.5
    ponder_prior_lambda: float = 0.2
    ponder_step_cost: float = 0.0
    ponder_kl_weight: float = 0.01
    ponder_entropy_weight: float = 0.0
    halt_correctness_weight: float = 0.05
    disable_halting: bool = False

    # NextLat-style predictive belief-state objective.
    nextlat_enabled: bool = True
    nextlat_weight: float = 0.1
    nextlat_loss_type: str = "smooth_l1"
    tapr_distance_mode: str = "hilbert"
    fw_bias: float = 0.5

    # Optional tropical polynomial chart transition after each recurrent block.
    chart_transition: bool = True
    chart_count: int = 4
    chart_transition_strength: float = 0.25


class TAPRBlock(nn.Module):
    def __init__(self, config: TAPRConfig) -> None:
        super().__init__()
        if config.tropical_attention_version == "v1":
            attention_cls = TropicalAttention
        elif config.tropical_attention_version == "v3":
            attention_cls = TropicalAttentionV3
        elif config.tropical_attention_version == "v4":
            attention_cls = TropicalAttentionV4
        else:
            raise ValueError("tropical_attention_version must be one of: 'v1', 'v3', 'v4'.")

        self.self_attn = attention_cls(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
            tropical_proj=config.tropical_proj,
            tropical_qkv_proj=config.tropical_qkv_proj,
            tropical_norm=config.tropical_norm,
            q_dropout=config.q_dropout,
            k_dropout=config.k_dropout,
            v_dropout=config.v_dropout,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


class NextLatPredictor(nn.Module):
    def __init__(self, config: TAPRConfig) -> None:
        super().__init__()
        self.up = CastedLinear(config.hidden_size, config.hidden_size * 2, bias=True)
        self.down = CastedLinear(config.hidden_size * 2, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(hidden_states)))


class TropicalChartTransition(nn.Module):
    """A compact tropical-polynomial state transition.

    Each chart is an affine map. The max over chart values is the tropical
    polynomial envelope; the active chart margin is logged as a routing signal.
    """

    def __init__(self, config: TAPRConfig) -> None:
        super().__init__()
        self.strength = config.chart_transition_strength
        self.norm_eps = config.rms_norm_eps
        self.chart_count = config.chart_count
        self.router = CastedLinear(config.hidden_size, self.chart_count, bias=True)
        self.charts = nn.ModuleList(
            [CastedLinear(config.hidden_size, config.hidden_size, bias=True) for _ in range(self.chart_count)]
        )
        self.gate = CastedLinear(config.hidden_size * 2, config.hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pooled = hidden_states.mean(dim=1)
        chart_scores = self.router(pooled).to(torch.float32)
        chart_values = torch.stack([chart(hidden_states) for chart in self.charts], dim=2)
        chart_values = chart_values + chart_scores.to(hidden_states.dtype)[:, None, :, None]
        update = chart_values.amax(dim=2)
        gate = torch.sigmoid(self.gate(torch.cat([hidden_states, update], dim=-1)))
        hidden_states = rms_norm(
            hidden_states + self.strength * gate * update,
            variance_epsilon=self.norm_eps,
        )

        topk = chart_scores.topk(k=min(2, self.chart_count), dim=-1).values
        if topk.shape[-1] == 1:
            margin = torch.zeros_like(topk[..., 0])
        else:
            margin = topk[..., 0] - topk[..., 1]
        return hidden_states, {
            "chart_margin": margin,
            "chart_gate": gate.mean(dim=(1, 2)),
        }


class TAPRInner(nn.Module):
    def __init__(self, config: TAPRConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.halt_head = CastedLinear(self.config.hidden_size + 4, 2, bias=True)
        self.nextlat = NextLatPredictor(self.config) if self.config.nextlat_enabled else None
        self.chart_transition = TropicalChartTransition(self.config) if self.config.chart_transition else None
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        self.pos_encoding_mode = self._resolve_pos_encoding_mode(self.config.pos_encodings)

        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.pos_encoding_mode == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        elif self.pos_encoding_mode == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )

        self.layers = nn.ModuleList([TAPRBlock(self.config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.halt_head.weight.zero_()
            if self.halt_head.bias is not None:
                self.halt_head.bias[0] = _logit(self.config.ponder_prior_lambda)
                self.halt_head.bias[1] = -_logit(self.config.ponder_prior_lambda)

        self.activation_probe = None

    @staticmethod
    def _resolve_pos_encoding_mode(pos_encodings: str) -> str:
        aliases = {
            "learnable": "learned",
        }
        mode = aliases.get(pos_encodings, pos_encodings)
        if mode not in {"learned", "rope", "none"}:
            raise ValueError("pos_encodings must be one of: 'learned', 'learnable', 'rope', 'none'.")
        return mode

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2,
            )
        if self.pos_encoding_mode == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int, device: Optional[torch.device] = None) -> TAPRCarry:
        if device is None:
            device = self.init_hidden.device
        return TAPRCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TAPRCarry) -> TAPRCarry:
        device = carry.current_hidden.device
        reset_flag = reset_flag.to(device)
        init_hidden = self.init_hidden.to(device).view(1, 1, -1)
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            init_hidden,
            carry.current_hidden,
        )
        return replace(carry, current_hidden=new_hidden)

    def _apply_reasoning_cycle(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, Optional[CosSin]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states = hidden_states + input_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        chart_stats: Dict[str, torch.Tensor] = {}
        if self.chart_transition is not None:
            hidden_states, chart_stats = self.chart_transition(hidden_states)
        return hidden_states, chart_stats

    def _nextlat_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.nextlat_loss_type == "mse":
            return (prediction.to(torch.float32) - target.detach().to(torch.float32)).pow(2).mean(dim=(1, 2))
        if self.config.nextlat_loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                prediction.to(torch.float32),
                target.detach().to(torch.float32),
                reduction="none",
            ).mean(dim=(1, 2))
        if self.config.nextlat_loss_type == "projective":
            dist = generalized_tropical_distance(
                prediction.to(torch.float32),
                target.detach().to(torch.float32),
                mode=self.config.tapr_distance_mode,
                bias=self.config.fw_bias,
            )
            return dist.mean(dim=1)
        raise ValueError("nextlat_loss_type must be one of: mse, smooth_l1, projective.")

    def forward(
        self,
        carry: TAPRCarry,
        batch: Dict[str, torch.Tensor],
        *,
        step_fraction: torch.Tensor,
        prev_logits: torch.Tensor,
    ) -> Tuple[TAPRCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.pos_encoding_mode == "rope":
            seq_info = dict(cos_sin=self.rotary_emb())
        else:
            seq_info = dict(cos_sin=None)

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states, _ = self._apply_reasoning_cycle(hidden_states, input_embeddings, seq_info)

        nextlat_source = hidden_states + input_embeddings
        nextlat_prediction = self.nextlat(nextlat_source) if self.nextlat is not None else None

        chart_stats: Dict[str, torch.Tensor] = {}
        for _ in range(self.config.L_cycles):
            hidden_states, chart_stats = self._apply_reasoning_cycle(hidden_states, input_embeddings, seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]

        if nextlat_prediction is None:
            nextlat_loss = torch.zeros(hidden_states.shape[0], dtype=torch.float32, device=hidden_states.device)
        else:
            nextlat_loss = self._nextlat_loss(nextlat_prediction, hidden_states)

        source_for_delta = nextlat_source.detach()
        state_delta = generalized_tropical_distance(
            hidden_states.to(torch.float32),
            source_for_delta.to(torch.float32),
            mode=self.config.tapr_distance_mode,
            bias=self.config.fw_bias,
        ).mean(dim=1)
        logit_delta = (output.detach().to(torch.float32) - prev_logits.to(torch.float32)).abs().mean(dim=(1, 2))

        halt_features = torch.cat(
            [
                hidden_states[:, 0].to(torch.float32),
                nextlat_loss.unsqueeze(-1),
                state_delta.unsqueeze(-1),
                logit_delta.unsqueeze(-1),
                step_fraction.to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )
        q_logits = self.halt_head(halt_features.to(hidden_states.dtype)).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        aux = {
            "nextlat_loss": nextlat_loss,
            "state_delta": state_delta,
            "logit_delta": logit_delta,
            **chart_stats,
        }

        probe = getattr(self, "activation_probe", None)
        if probe is not None:
            probe.record_tensor("hidden_states_final", hidden_states)
            probe.record_tensor("new_carry_current_hidden", new_carry.current_hidden)
            probe.record_tensor("output_logits", output)
            probe.record_tensor("q_logits_raw", q_logits)
            probe.record_tensor("q_halt_logits", q_halt_logits)
            probe.record_tensor("q_continue_logits", q_continue_logits)
            probe.record_tensor("tapr_nextlat_loss", nextlat_loss)
            probe.record_tensor("tapr_state_delta", state_delta)

        return new_carry, output, (q_halt_logits, q_continue_logits), aux


class TAPR(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TAPRConfig(**config_dict)
        self.inner = TAPRInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TAPRCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        base = self.inner.empty_carry(batch_size, device=device)
        logits_shape = (batch_size, self.config.seq_len, self.config.vocab_size)
        return TAPRCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            ponder_alive=torch.ones((batch_size,), dtype=torch.float32, device=device),
            ponder_mass=torch.zeros((batch_size,), dtype=torch.float32, device=device),
            ponder_expected_steps=torch.zeros((batch_size,), dtype=torch.float32, device=device),
            prev_logits=torch.zeros(logits_shape, dtype=torch.float32, device=device),
        )

    def forward(
        self,
        carry: TAPRCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[TAPRCarry, Dict[str, torch.Tensor]]:
        del compute_target_q

        if carry.halted is None or carry.steps is None or carry.current_data is None:
            raise ValueError("TAPR carry must be created with initial_carry().")

        new_carry = self.inner.reset_carry(carry.halted, carry)
        device = new_carry.current_hidden.device
        reset_flag = carry.halted.to(device)

        previous_steps = torch.where(reset_flag, torch.zeros_like(carry.steps), carry.steps.to(device))
        new_steps = previous_steps + 1
        current_data = {
            k: torch.where(
                reset_flag.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v.to(device),
            )
            for k, v in carry.current_data.items()
        }

        alive_before = torch.where(reset_flag, torch.ones_like(carry.ponder_alive), carry.ponder_alive.to(device))
        mass_before = torch.where(reset_flag, torch.zeros_like(carry.ponder_mass), carry.ponder_mass.to(device))
        expected_before = torch.where(
            reset_flag,
            torch.zeros_like(carry.ponder_expected_steps),
            carry.ponder_expected_steps.to(device),
        )
        prev_logits = torch.where(
            reset_flag.view(-1, 1, 1),
            torch.zeros_like(carry.prev_logits).to(device),
            carry.prev_logits.to(device),
        )

        step_fraction = new_steps.to(torch.float32) / max(self.config.loops, 1)
        new_carry, logits, (q_halt_logits, q_continue_logits), aux = self.inner(
            new_carry,
            current_data,
            step_fraction=step_fraction,
            prev_logits=prev_logits,
        )

        halt_prob = torch.sigmoid(q_halt_logits)
        min_steps = max(1, min(self.config.ponder_min_steps, self.config.loops))
        before_min = new_steps < min_steps
        is_last_step = new_steps >= self.config.loops

        if (not self.config.ponder_enabled) or self.config.disable_halting:
            effective_halt_prob = torch.where(is_last_step, torch.ones_like(halt_prob), torch.zeros_like(halt_prob))
        else:
            effective_halt_prob = halt_prob
            effective_halt_prob = torch.where(before_min, torch.zeros_like(effective_halt_prob), effective_halt_prob)
            effective_halt_prob = torch.where(is_last_step, torch.ones_like(effective_halt_prob), effective_halt_prob)

        ponder_prob = alive_before * effective_halt_prob
        alive_after = (alive_before * (1.0 - effective_halt_prob)).clamp_min(0.0)
        mass_after = (mass_before + ponder_prob).clamp_max(1.0)
        expected_after = expected_before + ponder_prob * new_steps.to(torch.float32)

        if self.training:
            # Train the full PonderNet distribution until the forced final step.
            halted = is_last_step
        elif (not self.config.ponder_enabled) or self.config.disable_halting:
            halted = is_last_step
        else:
            deterministic_halt = halt_prob >= self.config.ponder_eval_threshold
            halted = (deterministic_halt & (~before_min)) | is_last_step

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "halt_prob": halt_prob,
            "effective_halt_prob": effective_halt_prob,
            "ponder_prob": ponder_prob,
            "ponder_alive_before": alive_before,
            "ponder_alive_after": alive_after,
            "ponder_mass": mass_after,
            "ponder_expected_steps": expected_after,
            "is_last_step": is_last_step,
            "nextlat_loss": aux["nextlat_loss"],
            "state_delta": aux["state_delta"],
            "logit_delta": aux["logit_delta"],
        }
        if "chart_margin" in aux:
            outputs["chart_margin"] = aux["chart_margin"]
        if "chart_gate" in aux:
            outputs["chart_gate"] = aux["chart_gate"]

        return (
            TAPRCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=current_data,
                ponder_alive=alive_after.detach(),
                ponder_mass=mass_after.detach(),
                ponder_expected_steps=expected_after.detach(),
                prev_logits=logits.detach().to(torch.float32),
            ),
            outputs,
        )


class TAPRLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = {
            "stablemax_cross_entropy": stablemax_cross_entropy,
            "softmax_cross_entropy": softmax_cross_entropy,
        }[loss_type]
        self.config = getattr(model, "config", None)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def _prior_prob(self, steps: torch.Tensor, is_last_step: torch.Tensor) -> torch.Tensor:
        prior_lambda = getattr(self.config, "ponder_prior_lambda", 0.2)
        prior_lambda = min(max(float(prior_lambda), 1e-5), 1.0 - 1e-5)
        step_f = steps.to(torch.float32).clamp_min(1.0)
        prior = prior_lambda * torch.pow(torch.as_tensor(1.0 - prior_lambda, device=steps.device), step_f - 1.0)
        tail = torch.pow(torch.as_tensor(1.0 - prior_lambda, device=steps.device), step_f - 1.0)
        return torch.where(is_last_step, tail, prior).clamp_min(1e-8)

    def forward(
        self,
        return_keys: Set[str],
        return_raw_outputs: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        logits = outputs["logits"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1)
            is_correct = mask & (outputs["preds"] == labels)
            cell_acc = (is_correct.to(torch.float32).sum(-1) / loss_divisor).to(torch.float32)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_examples = loss_counts > 0
            count = valid_examples.sum().clamp_min(1)

        token_loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID)
        per_example_task_loss = token_loss.sum(-1) / loss_divisor
        ponder_prob = outputs["ponder_prob"].to(torch.float32)
        alive_before = outputs["ponder_alive_before"].to(torch.float32)
        task_loss = (ponder_prob * per_example_task_loss).sum()

        nextlat_vec = outputs["nextlat_loss"].to(torch.float32)
        nextlat_loss = (alive_before * nextlat_vec).sum()

        prior = self._prior_prob(new_carry.steps, outputs["is_last_step"]).to(torch.float32)
        ponder_kl = (
            ponder_prob.clamp_min(1e-8)
            * (ponder_prob.clamp_min(1e-8).log() - prior.log())
        ).sum()
        ponder_entropy = -(ponder_prob.clamp_min(1e-8) * ponder_prob.clamp_min(1e-8).log()).sum()
        expected_step_loss = (ponder_prob * new_carry.steps.to(torch.float32)).sum()

        halt_correctness_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"].to(torch.float32),
            seq_is_correct.to(torch.float32),
            reduction="none",
        )
        halt_correctness_loss = (alive_before * halt_correctness_loss).sum()

        nextlat_weight = getattr(self.config, "nextlat_weight", 0.0)
        ponder_kl_weight = getattr(self.config, "ponder_kl_weight", 0.0)
        ponder_step_cost = getattr(self.config, "ponder_step_cost", 0.0)
        ponder_entropy_weight = getattr(self.config, "ponder_entropy_weight", 0.0)
        halt_correctness_weight = getattr(self.config, "halt_correctness_weight", 0.0)

        total_loss = (
            task_loss
            + nextlat_weight * nextlat_loss
            + ponder_kl_weight * ponder_kl
            + ponder_step_cost * expected_step_loss
            - ponder_entropy_weight * ponder_entropy
            + halt_correctness_weight * halt_correctness_loss
        )

        halt_pred = outputs["halt_prob"] >= getattr(self.config, "ponder_eval_threshold", 0.5)
        over_halt = halt_pred & (~seq_is_correct) & valid_examples
        under_halt = (~halt_pred) & seq_is_correct & valid_examples
        steps_f = new_carry.steps.to(torch.float32).clamp_min(1.0)

        metrics: Dict[str, torch.Tensor] = {
            "count": count.to(torch.float32),
            "accuracy": torch.where(valid_examples, cell_acc, torch.zeros_like(cell_acc)).sum(),
            "exact_accuracy": (valid_examples & seq_is_correct).sum().to(torch.float32),
            "accuracy_per_step": torch.where(valid_examples, cell_acc / steps_f, torch.zeros_like(cell_acc)).sum(),
            "steps": torch.where(valid_examples, new_carry.steps, torch.zeros_like(new_carry.steps)).sum().to(torch.float32),
            "ponder_expected_steps": torch.where(
                valid_examples,
                outputs["ponder_expected_steps"].to(torch.float32),
                torch.zeros_like(outputs["ponder_expected_steps"].to(torch.float32)),
            ).sum(),
            "ponder_prob": torch.where(valid_examples, ponder_prob, torch.zeros_like(ponder_prob)).sum(),
            "ponder_mass": torch.where(
                valid_examples,
                outputs["ponder_mass"].to(torch.float32),
                torch.zeros_like(outputs["ponder_mass"].to(torch.float32)),
            ).sum(),
            "ponder_alive": torch.where(valid_examples, outputs["ponder_alive_after"], torch.zeros_like(ponder_prob)).sum(),
            "halt_prob": torch.where(valid_examples, outputs["halt_prob"], torch.zeros_like(ponder_prob)).sum(),
            "over_halt_rate": over_halt.sum().to(torch.float32),
            "under_halt_rate": under_halt.sum().to(torch.float32),
            "halt_brier": torch.where(
                valid_examples,
                (outputs["halt_prob"].to(torch.float32) - seq_is_correct.to(torch.float32)).pow(2),
                torch.zeros_like(ponder_prob),
            ).sum(),
            "lm_loss": task_loss.detach(),
            "nextlat_loss": nextlat_loss.detach(),
            "ponder_kl_loss": ponder_kl.detach(),
            "ponder_step_loss": expected_step_loss.detach(),
            "ponder_entropy": ponder_entropy.detach(),
            "halt_correctness_loss": halt_correctness_loss.detach(),
            "state_delta": torch.where(valid_examples, outputs["state_delta"], torch.zeros_like(ponder_prob)).sum(),
            "logit_delta": torch.where(valid_examples, outputs["logit_delta"], torch.zeros_like(ponder_prob)).sum(),
        }
        if "chart_margin" in outputs:
            metrics["chart_margin"] = torch.where(
                valid_examples,
                outputs["chart_margin"].to(torch.float32),
                torch.zeros_like(ponder_prob),
            ).sum()
        if "chart_gate" in outputs:
            metrics["chart_gate"] = torch.where(
                valid_examples,
                outputs["chart_gate"].to(torch.float32),
                torch.zeros_like(ponder_prob),
            ).sum()

        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = outputs

        for key in return_keys:
            if key in outputs:
                returned_outputs[key] = outputs[key].detach()

        return (
            new_carry,
            total_loss,
            metrics,
            returned_outputs,
            new_carry.halted.all(),
        )


# Backward-compatible aliases for older import strings while the new config uses TAPR.
TARMCarry = TAPRCarry
TARMConfig = TAPRConfig
TARMBlock = TAPRBlock
TARM_Inner = TAPRInner
TARM = TAPR
