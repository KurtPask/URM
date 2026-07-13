from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Set, Tuple
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
    ToricBoundaryRouter,
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
    tapr_architecture: str = "legacy"

    # PonderNet-style adaptive compute.
    ponder_enabled: bool = True
    ponder_min_steps: int = 2
    readout_every: int = 1
    ponder_eval_threshold: float = 0.5
    ponder_eval_mode: str = "threshold"
    ponder_eval_cdf_threshold: float = 0.5
    ponder_prior_lambda: float = 0.2
    ponder_step_cost: float = 0.0
    ponder_kl_weight: float = 0.01
    ponder_entropy_weight: float = 0.0
    halt_correctness_weight: float = 0.0
    disable_halting: bool = False
    # -1 preserves the historical detach_recurrent_state flag; 0 means full
    # BPTT; k > 0 detaches recurrent and ponder state every k steps.
    bptt_window: int = -1
    detach_recurrent_state: bool = True
    detach_ponder_state: bool = True
    eval_halt_mode: str = "threshold"
    halt_head_type: str = "q_pair"
    ponder_ce_weight: float = 1.0
    final_ce_weight: float = 0.0

    # Solution quality is deliberately separate from the PonderNet stop head.
    # ARC may use this score to rank augmented candidate outputs.
    solution_score_enabled: bool = False
    solution_score_weight: float = 0.0
    solution_score_normalize_to_ce: bool = True
    log_train_step_metrics: bool = False
    log_eval_step_metrics: bool = True

    # NextLat-style predictive belief-state objective.
    nextlat_enabled: bool = True
    nextlat_weight: float = 0.1
    nextlat_loss_type: str = "smooth_l1"
    nextlat_supervised_only: bool = False
    nextlat_normalize_to_ce: bool = False
    tapr_distance_mode: str = "hilbert"
    fw_bias: float = 0.5
    aux_normalize_eps: float = 1e-6
    aux_normalize_max_scale: float = 10.0

    # Optional task-specific consistency residuals.
    task_residual_type: str = "none"
    task_residual_weight: float = 0.0
    task_residual_normalize_to_ce: bool = True
    task_residual_grid_size: int = 0
    task_residual_clrs_nodes: int = 0
    task_residual_clrs_max_weight: int = 9
    task_residual_clrs_dist_cap: int = 32
    task_residual_tau: float = 1.0

    # Optional tropical polynomial chart transition after each recurrent block.
    chart_transition: bool = True
    chart_count: int = 4
    chart_transition_strength: float = 0.25

    # Optional toric-compactification boundary router around the whole block.
    toric_router: bool = False
    toric_route_dim: int = 4
    toric_gate_tau: float = 1.0
    toric_gate_temp: float = 0.5
    toric_dir_tau: float = 0.25
    toric_dir_temp: float = 0.1
    toric_learn_thresholds: bool = False
    toric_gate_mode: str = "calibrated"
    toric_gate_quantile: float = 0.95
    toric_gate_ema: float = 0.05
    toric_gate_stat: str = "route_norm"
    toric_pinv_ridge: float = 1e-3
    toric_partition_mode: str = "hard"


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
        self.toric_router = None
        if config.toric_router:
            self.toric_router = ToricBoundaryRouter(
                hidden_size=config.hidden_size,
                route_dim=config.toric_route_dim,
                gate_tau=config.toric_gate_tau,
                gate_temp=config.toric_gate_temp,
                dir_tau=config.toric_dir_tau,
                dir_temp=config.toric_dir_temp,
                learn_thresholds=config.toric_learn_thresholds,
                gate_mode=config.toric_gate_mode,
                gate_quantile=config.toric_gate_quantile,
                gate_ema=config.toric_gate_ema,
                gate_stat=config.toric_gate_stat,
                pinv_ridge=config.toric_pinv_ridge,
                partition_mode=config.toric_partition_mode,
            )

        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def _forward_core(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        if self.toric_router is None:
            return self._forward_core(cos_sin, hidden_states)

        x_quot, alpha, fan_state = self.toric_router.project_input(hidden_states)
        interior = self._forward_core(cos_sin, hidden_states)
        boundary = self._forward_core(cos_sin, x_quot)
        boundary = self.toric_router.project_output(boundary, fan_state)
        return self.toric_router.blend(interior, boundary, alpha)


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
        if self.config.tapr_architecture == "clean_full":
            if self.config.H_cycles != 1:
                raise ValueError("clean_full TAPR uses one recurrence clock: H_cycles must be 1.")
            if self.config.L_cycles != 1:
                raise ValueError("clean_full TAPR uses one recurrence clock: L_cycles must be 1.")
            if self.config.halt_head_type != "scalar":
                raise ValueError("clean_full TAPR uses a scalar PonderNet halt head.")
            if self.config.detach_recurrent_state:
                raise ValueError("clean_full TAPR requires detach_recurrent_state=false.")
            if self.config.detach_ponder_state:
                raise ValueError("clean_full TAPR requires detach_ponder_state=false.")
            if self.config.bptt_window < 0:
                raise ValueError("clean_full TAPR requires bptt_window>=0 (0 means full BPTT).")
            if self.config.halt_correctness_weight != 0:
                raise ValueError("clean_full TAPR cannot combine PonderNet with halt_correctness_weight.")
            if not self.config.ponder_enabled or self.config.disable_halting:
                raise ValueError("clean_full TAPR requires PonderNet halting to be enabled.")
            if self.config.nextlat_weight > 0 and not self.config.nextlat_enabled:
                raise ValueError("nextlat_weight requires nextlat_enabled=true.")
            if self.config.nextlat_enabled and self.config.nextlat_weight > 0:
                if not self.config.nextlat_supervised_only:
                    raise ValueError("clean_full NextLat must use supervised/decision tokens only.")
                if not self.config.nextlat_normalize_to_ce:
                    raise ValueError("clean_full NextLat must be normalized to the CE magnitude.")
            if self.config.solution_score_weight > 0 and not self.config.solution_score_enabled:
                raise ValueError("solution_score_weight requires solution_score_enabled=true.")
            if self.config.ponder_min_steps < 1 or self.config.ponder_min_steps > self.config.loops:
                raise ValueError("ponder_min_steps must be in [1, loops].")
            if self.config.readout_every < 1:
                raise ValueError("readout_every must be >= 1.")
            if self.config.eval_halt_mode not in {"threshold", "cdf", "full"}:
                raise ValueError("eval_halt_mode must be: threshold, cdf, or full.")
            if not 0.0 < self.config.ponder_prior_lambda < 1.0:
                raise ValueError("ponder_prior_lambda must be strictly between 0 and 1.")
            if not 0.0 < self.config.ponder_eval_cdf_threshold <= 1.0:
                raise ValueError("ponder_eval_cdf_threshold must be in (0, 1].")
        if self.config.halt_head_type not in {"q_pair", "scalar"}:
            raise ValueError("halt_head_type must be either 'q_pair' or 'scalar'.")
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
        halt_out_features = 1 if self.config.halt_head_type == "scalar" else 2
        self.halt_head = CastedLinear(self.config.hidden_size + 4, halt_out_features, bias=True)
        self.solution_score_head = (
            CastedLinear(self.config.hidden_size + 4, 1, bias=True)
            if self.config.solution_score_enabled
            else None
        )
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
                if self.config.halt_head_type == "q_pair":
                    self.halt_head.bias[1] = -_logit(self.config.ponder_prior_lambda)
            if self.solution_score_head is not None:
                self.solution_score_head.weight.zero_()
                if self.solution_score_head.bias is not None:
                    self.solution_score_head.bias.zero_()

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

    def _nextlat_token_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.nextlat_loss_type == "mse":
            return (prediction.to(torch.float32) - target.detach().to(torch.float32)).pow(2).mean(dim=-1)
        if self.config.nextlat_loss_type == "smooth_l1":
            return F.smooth_l1_loss(
                prediction.to(torch.float32),
                target.detach().to(torch.float32),
                reduction="none",
            ).mean(dim=-1)
        if self.config.nextlat_loss_type == "projective":
            return generalized_tropical_distance(
                prediction.to(torch.float32),
                target.detach().to(torch.float32),
                mode=self.config.tapr_distance_mode,
                bias=self.config.fw_bias,
            )
        raise ValueError("nextlat_loss_type must be one of: mse, smooth_l1, projective.")

    def _nextlat_loss(self, prediction: torch.Tensor, target: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        token_loss = self._nextlat_token_loss(prediction, target)
        if self.config.nextlat_supervised_only and "labels" in batch:
            seq_mask = batch["labels"].ne(IGNORE_LABEL_ID)
            if self.puzzle_emb_len > 0:
                prefix_mask = torch.zeros(
                    seq_mask.shape[0],
                    self.puzzle_emb_len,
                    dtype=torch.bool,
                    device=seq_mask.device,
                )
                full_mask = torch.cat([prefix_mask, seq_mask], dim=1)
            else:
                full_mask = seq_mask
            full_mask = full_mask[:, : token_loss.shape[1]]
            denom = full_mask.sum(dim=1).clamp_min(1).to(torch.float32)
            return (token_loss * full_mask.to(torch.float32)).sum(dim=1) / denom
        return token_loss.mean(dim=1)

    def forward(
        self,
        carry: TAPRCarry,
        batch: Dict[str, torch.Tensor],
        *,
        step_fraction: torch.Tensor,
        prev_logits: torch.Tensor,
    ) -> Tuple[TAPRCarry, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if self.pos_encoding_mode == "rope":
            seq_info = dict(cos_sin=self.rotary_emb())
        else:
            seq_info = dict(cos_sin=None)

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        hidden_states = carry.current_hidden
        if self.config.tapr_architecture != "clean_full" and self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states, _ = self._apply_reasoning_cycle(hidden_states, input_embeddings, seq_info)

        nextlat_source = hidden_states + input_embeddings
        nextlat_prediction = self.nextlat(nextlat_source) if self.nextlat is not None else None

        chart_stats: Dict[str, torch.Tensor] = {}
        for _ in range(self.config.L_cycles):
            hidden_states, chart_stats = self._apply_reasoning_cycle(hidden_states, input_embeddings, seq_info)

        legacy_detach = self.config.detach_recurrent_state and self.config.bptt_window < 0
        carry_hidden = hidden_states.detach() if legacy_detach else hidden_states
        new_carry = replace(carry, current_hidden=carry_hidden)
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]

        if nextlat_prediction is None:
            nextlat_loss = torch.zeros(hidden_states.shape[0], dtype=torch.float32, device=hidden_states.device)
        else:
            nextlat_loss = self._nextlat_loss(nextlat_prediction, hidden_states, batch)

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
        halt_raw = self.halt_head(halt_features.to(hidden_states.dtype)).to(torch.float32)
        if self.config.halt_head_type == "scalar":
            halt_logit = halt_raw.squeeze(-1)
            q_continue_logits = None
        else:
            halt_logit = halt_raw[..., 0]
            q_continue_logits = halt_raw[..., 1]

        solution_score_logits = None
        if self.solution_score_head is not None:
            solution_score_logits = self.solution_score_head(
                halt_features.to(hidden_states.dtype)
            ).squeeze(-1).to(torch.float32)

        aux = {
            "nextlat_loss": nextlat_loss,
            "state_delta": state_delta,
            "logit_delta": logit_delta,
            **chart_stats,
        }
        if solution_score_logits is not None:
            aux["solution_score_logits"] = solution_score_logits

        probe = getattr(self, "activation_probe", None)
        if probe is not None:
            probe.record_tensor("hidden_states_final", hidden_states)
            probe.record_tensor("new_carry_current_hidden", new_carry.current_hidden)
            probe.record_tensor("output_logits", output)
            probe.record_tensor("halt_logits_raw", halt_raw)
            probe.record_tensor("halt_logit", halt_logit)
            if q_continue_logits is not None:
                probe.record_tensor("q_continue_logits", q_continue_logits)
            if solution_score_logits is not None:
                probe.record_tensor("solution_score_logits", solution_score_logits)
            probe.record_tensor("tapr_nextlat_loss", nextlat_loss)
            probe.record_tensor("tapr_state_delta", state_delta)

        return new_carry, output, halt_logit, q_continue_logits, aux


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
        force_full_trajectory: bool = False,
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
        new_carry, logits, halt_logit, q_continue_logits, aux = self.inner(
            new_carry,
            current_data,
            step_fraction=step_fraction,
            prev_logits=prev_logits,
        )

        halt_prob = torch.sigmoid(halt_logit)
        min_steps = max(1, min(self.config.ponder_min_steps, self.config.loops))
        before_min = new_steps < min_steps
        is_last_step = new_steps >= self.config.loops
        readout_every = max(1, int(getattr(self.config, "readout_every", 1)))
        is_readout_step = (new_steps.remainder(readout_every) == 0) | is_last_step

        if (not self.config.ponder_enabled) or self.config.disable_halting:
            effective_halt_prob = torch.where(is_last_step, torch.ones_like(halt_prob), torch.zeros_like(halt_prob))
        else:
            effective_halt_prob = torch.where(is_readout_step, halt_prob, torch.zeros_like(halt_prob))
            effective_halt_prob = torch.where(before_min, torch.zeros_like(effective_halt_prob), effective_halt_prob)
            effective_halt_prob = torch.where(is_last_step, torch.ones_like(effective_halt_prob), effective_halt_prob)

        ponder_prob = alive_before * effective_halt_prob
        alive_after = (alive_before * (1.0 - effective_halt_prob)).clamp_min(0.0)
        mass_after = (mass_before + ponder_prob).clamp_max(1.0)
        expected_after = expected_before + ponder_prob * new_steps.to(torch.float32)

        if self.training or force_full_trajectory:
            # Train the full PonderNet distribution until the forced final step.
            halted = is_last_step
        elif (not self.config.ponder_enabled) or self.config.disable_halting:
            halted = is_last_step
        elif self.config.eval_halt_mode == "full":
            halted = is_last_step
        else:
            if self.config.eval_halt_mode == "threshold":
                deterministic_halt = halt_prob >= self.config.ponder_eval_threshold
            elif self.config.eval_halt_mode == "cdf":
                deterministic_halt = mass_after >= self.config.ponder_eval_cdf_threshold
            else:
                raise ValueError("eval_halt_mode must be: threshold, cdf, or full.")
            halted = (deterministic_halt & (~before_min)) | is_last_step

        outputs = {
            "logits": logits,
            "halt_logit": halt_logit,
            "q_halt_logits": halt_logit,
            "halt_prob": halt_prob,
            "effective_halt_prob": effective_halt_prob,
            "ponder_prob": ponder_prob,
            "ponder_alive_before": alive_before,
            "ponder_alive_after": alive_after,
            "ponder_mass": mass_after,
            "ponder_expected_steps": expected_after,
            "is_last_step": is_last_step,
            "is_readout_step": is_readout_step,
            "nextlat_loss": aux["nextlat_loss"],
            "state_delta": aux["state_delta"],
            "logit_delta": aux["logit_delta"],
        }
        if q_continue_logits is not None:
            outputs["q_continue_logits"] = q_continue_logits
        if "solution_score_logits" in aux:
            outputs["solution_score_logits"] = aux["solution_score_logits"]
        if "chart_margin" in aux:
            outputs["chart_margin"] = aux["chart_margin"]
        if "chart_gate" in aux:
            outputs["chart_gate"] = aux["chart_gate"]

        carry_hidden = new_carry.current_hidden
        if self.config.bptt_window == 0:
            carry_current_hidden = carry_hidden
        elif self.config.bptt_window > 0:
            detach_boundary = new_steps.remainder(self.config.bptt_window).eq(0)
            carry_current_hidden = torch.where(
                detach_boundary.view(-1, 1, 1),
                carry_hidden.detach(),
                carry_hidden,
            )
        else:
            carry_current_hidden = carry_hidden

        if self.config.tapr_architecture == "clean_full":
            # The survival product is cheap and must remain differentiable across
            # every recurrent window even when the hidden state is truncated.
            carry_ponder_alive = alive_after
            carry_ponder_mass = mass_after
            carry_ponder_expected_steps = expected_after
        elif self.config.bptt_window == 0:
            carry_ponder_alive = alive_after
            carry_ponder_mass = mass_after
            carry_ponder_expected_steps = expected_after
        elif self.config.bptt_window > 0:
            detach_boundary = new_steps.remainder(self.config.bptt_window).eq(0)
            carry_ponder_alive = torch.where(detach_boundary, alive_after.detach(), alive_after)
            carry_ponder_mass = torch.where(detach_boundary, mass_after.detach(), mass_after)
            carry_ponder_expected_steps = torch.where(detach_boundary, expected_after.detach(), expected_after)
        elif self.config.detach_ponder_state:
            carry_ponder_alive = alive_after.detach()
            carry_ponder_mass = mass_after.detach()
            carry_ponder_expected_steps = expected_after.detach()
        else:
            carry_ponder_alive = alive_after
            carry_ponder_mass = mass_after
            carry_ponder_expected_steps = expected_after

        return (
            TAPRCarry(
                current_hidden=carry_current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=current_data,
                ponder_alive=carry_ponder_alive,
                ponder_mass=carry_ponder_mass,
                ponder_expected_steps=carry_ponder_expected_steps,
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
        min_steps = max(1, min(int(getattr(self.config, "ponder_min_steps", 1)), self.config.loops))
        readout_every = max(1, int(getattr(self.config, "readout_every", 1)))
        first_decision = ((min_steps + readout_every - 1) // readout_every) * readout_every
        num_nonfinal_decisions = (
            0
            if first_decision >= self.config.loops
            else 1 + (self.config.loops - 1 - first_decision) // readout_every
        )

        step_i = steps.to(torch.long)
        decision_index = torch.div(
            (step_i - first_decision).clamp_min(0),
            readout_every,
            rounding_mode="floor",
        ).to(torch.float32)
        one_minus = torch.as_tensor(1.0 - prior_lambda, dtype=torch.float32, device=steps.device)
        prior = prior_lambda * torch.pow(one_minus, decision_index)
        tail = torch.pow(one_minus, num_nonfinal_decisions)
        valid_decision = (step_i >= first_decision) & (step_i.remainder(readout_every) == 0)
        prior = torch.where(valid_decision, prior, torch.zeros_like(prior))
        return torch.where(is_last_step, tail, prior)

    def _trajectory_prior(self, device: torch.device) -> torch.Tensor:
        """Normalized shifted geometric prior over actual halt opportunities."""
        loops = int(self.config.loops)
        min_steps = max(1, min(int(self.config.ponder_min_steps), loops))
        readout_every = max(1, int(self.config.readout_every))
        prior_lambda = min(max(float(self.config.ponder_prior_lambda), 1e-5), 1.0 - 1e-5)
        prior = torch.zeros(loops, dtype=torch.float32, device=device)
        decision_steps = [
            step
            for step in range(1, loops)
            if step >= min_steps and step % readout_every == 0
        ]
        one_minus = torch.as_tensor(1.0 - prior_lambda, dtype=torch.float32, device=device)
        for decision_index, step in enumerate(decision_steps):
            prior[step - 1] = prior_lambda * torch.pow(one_minus, decision_index)
        prior[-1] = torch.pow(one_minus, len(decision_steps))
        return prior

    def _normalize_aux(self, aux_loss: torch.Tensor, task_loss: torch.Tensor, enabled: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if not enabled:
            return aux_loss, torch.ones((), dtype=torch.float32, device=aux_loss.device)
        eps = float(getattr(self.config, "aux_normalize_eps", 1e-6))
        max_scale = float(getattr(self.config, "aux_normalize_max_scale", 10.0))
        raw_scale = (task_loss.detach() / aux_loss.detach().clamp_min(eps)).clamp(max=max_scale)
        scale = torch.where(
            aux_loss.detach() > eps,
            raw_scale,
            torch.ones((), dtype=torch.float32, device=aux_loss.device),
        )
        return aux_loss * scale, scale

    def _sudoku_residual(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] < 81 or logits.shape[-1] < 11:
            return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
        probs = torch.softmax(logits[:, :81].to(torch.float32), dim=-1)[..., 2:11]
        grid = probs.reshape(logits.shape[0], 9, 9, 9)
        row_loss = (grid.sum(dim=2) - 1.0).pow(2).mean(dim=(1, 2))
        col_loss = (grid.sum(dim=1) - 1.0).pow(2).mean(dim=(1, 2))
        boxes = grid.reshape(logits.shape[0], 3, 3, 3, 3, 9).sum(dim=(2, 4))
        box_loss = (boxes - 1.0).pow(2).mean(dim=(1, 2, 3))
        return (row_loss + col_loss + box_loss) / 3.0

    def _maze_residual(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        grid_size = int(getattr(self.config, "task_residual_grid_size", 0))
        if grid_size <= 0:
            grid_size = int(math.isqrt(logits.shape[1]))
        if grid_size * grid_size != logits.shape[1] or logits.shape[-1] <= 5:
            return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)

        inputs = batch["inputs"][:, : logits.shape[1]]
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        path_prob = probs[..., 5].reshape(-1, grid_size, grid_size)
        wall = inputs.eq(1).reshape(-1, grid_size, grid_size)
        nonwall = (~wall).to(torch.float32)
        start_or_goal = (inputs.eq(3) | inputs.eq(4)).reshape(-1, grid_size, grid_size).to(torch.float32)
        certificate = torch.maximum(path_prob, start_or_goal)

        zero_row = torch.zeros_like(certificate[:, :1, :])
        zero_col = torch.zeros_like(certificate[:, :, :1])
        up = torch.cat([zero_row, certificate[:, :-1, :]], dim=1)
        down = torch.cat([certificate[:, 1:, :], zero_row], dim=1)
        left = torch.cat([zero_col, certificate[:, :, :-1]], dim=2)
        right = torch.cat([certificate[:, :, 1:], zero_col], dim=2)
        neighbor = torch.stack([up, down, left, right], dim=0).amax(dim=0)

        wall_loss = (path_prob * wall.to(torch.float32)).mean(dim=(1, 2))
        path_mass = (path_prob * nonwall).sum(dim=(1, 2)).detach().clamp_min(1.0)
        continuity = (
            path_prob
            * nonwall
            * F.relu(0.5 - neighbor).pow(2)
        ).sum(dim=(1, 2)) / path_mass
        endpoint_count = start_or_goal.sum(dim=(1, 2)).clamp_min(1.0)
        endpoint_support = (
            start_or_goal
            * F.relu(0.25 - neighbor).pow(2)
        ).sum(dim=(1, 2)) / endpoint_count
        return wall_loss + continuity + endpoint_support

    def _clrs_bellman_residual(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = int(getattr(self.config, "task_residual_clrs_nodes", 0))
        max_weight = int(getattr(self.config, "task_residual_clrs_max_weight", 9))
        dist_cap = int(getattr(self.config, "task_residual_clrs_dist_cap", 32))
        if nodes <= 0 or logits.shape[1] < nodes + nodes * nodes:
            return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)

        weight_base = 5
        dist_base = weight_base + max_weight
        if logits.shape[-1] <= dist_base:
            return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)

        max_token = min(logits.shape[-1], dist_base + dist_cap + 1)
        dist_tokens = torch.arange(dist_base, max_token, device=logits.device)
        dist_values = (dist_tokens - dist_base).to(torch.float32)
        probs = torch.softmax(logits[:, :nodes].to(torch.float32), dim=-1)
        dist_probs = probs.index_select(dim=-1, index=dist_tokens)
        dist_mass = dist_probs.sum(dim=-1).clamp_min(1e-6)
        pred_dist = (dist_probs * dist_values.view(1, 1, -1)).sum(dim=-1) / dist_mass

        inputs = batch["inputs"]
        source_mask = inputs[:, :nodes].eq(2)
        matrix = inputs[:, nodes : nodes + nodes * nodes].reshape(-1, nodes, nodes)
        edge_mask = (matrix >= weight_base) & (matrix < weight_base + max_weight)
        weights = (matrix.to(torch.float32) - float(weight_base) + 1.0).clamp_min(0.0)

        pred_u = pred_dist.unsqueeze(2)
        pred_v = pred_dist.unsqueeze(1)
        edge_f = edge_mask.to(torch.float32)
        edge_count = edge_f.sum(dim=(1, 2)).clamp_min(1.0)
        upper = (F.relu(pred_v - pred_u - weights).pow(2) * edge_f).sum(dim=(1, 2)) / edge_count

        tau = max(float(getattr(self.config, "task_residual_tau", 1.0)), 1e-3)
        large = torch.full_like(weights, 1e4)
        candidates = torch.where(edge_mask, pred_u + weights, large)
        smooth_min = -tau * torch.logsumexp(-candidates / tau, dim=1)
        non_source = (~source_mask).to(torch.float32)
        lower = (F.relu(smooth_min - pred_dist).pow(2) * non_source).sum(dim=1) / non_source.sum(dim=1).clamp_min(1.0)
        source = (pred_dist.pow(2) * source_mask.to(torch.float32)).sum(dim=1) / source_mask.sum(dim=1).clamp_min(1.0)
        mass = (1.0 - dist_mass).pow(2).mean(dim=1)
        return upper + lower + source + 0.1 * mass

    def _task_residual_vec(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        residual_type = getattr(self.config, "task_residual_type", "none")
        if residual_type in {"none", "", None}:
            return torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
        if residual_type == "sudoku_constraints":
            return self._sudoku_residual(logits)
        if residual_type == "maze_certificate":
            return self._maze_residual(logits, batch)
        if residual_type == "clrs_bellman":
            return self._clrs_bellman_residual(logits, batch)
        raise ValueError(f"Unknown task_residual_type: {residual_type}")

    def _trajectory_policy_indices(
        self,
        halt_probs: torch.Tensor,
        cumulative_mass: torch.Tensor,
    ) -> torch.Tensor:
        loops, batch_size = halt_probs.shape
        steps = torch.arange(1, loops + 1, device=halt_probs.device)
        allowed = steps.ge(int(self.config.ponder_min_steps)) & (
            steps.remainder(max(1, int(self.config.readout_every))).eq(0) | steps.eq(loops)
        )
        if self.config.eval_halt_mode in {"threshold", "full"}:
            triggered = halt_probs >= float(self.config.ponder_eval_threshold)
        elif self.config.eval_halt_mode == "cdf":
            triggered = cumulative_mass >= float(self.config.ponder_eval_cdf_threshold)
        else:
            raise ValueError("eval_halt_mode must be: threshold, cdf, or full.")
        if self.config.eval_halt_mode == "full":
            triggered = torch.zeros_like(triggered, dtype=torch.bool)
        triggered = triggered & allowed[:, None]
        triggered[-1] = True
        return triggered.to(torch.int64).argmax(dim=0).reshape(batch_size)

    def _forward_trajectory(
        self,
        return_keys: Set[str],
        return_raw_outputs: bool,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        if self.config.tapr_architecture != "clean_full":
            raise ValueError("Trajectory training is reserved for tapr_architecture=clean_full.")

        batch = model_kwargs["batch"]
        carry = self.initial_carry(batch)
        labels = batch["labels"]
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1)
        valid_examples = loss_counts > 0
        count = valid_examples.sum().clamp_min(1).to(torch.float32)

        ce_steps: List[torch.Tensor] = []
        nextlat_steps: List[torch.Tensor] = []
        residual_steps: List[torch.Tensor] = []
        ponder_steps: List[torch.Tensor] = []
        halt_steps: List[torch.Tensor] = []
        mass_steps: List[torch.Tensor] = []
        state_delta_steps: List[torch.Tensor] = []
        logit_delta_steps: List[torch.Tensor] = []
        cell_accuracy_steps: List[torch.Tensor] = []
        exact_accuracy_steps: List[torch.Tensor] = []
        prediction_steps: List[torch.Tensor] = []
        solution_score_steps: List[torch.Tensor] = []
        chart_margin_steps: List[torch.Tensor] = []
        chart_gate_steps: List[torch.Tensor] = []
        final_outputs: Dict[str, torch.Tensor] = {}

        for _ in range(int(self.config.loops)):
            carry, outputs = self.model(
                carry=carry,
                batch=batch,
                compute_target_q=False,
                force_full_trajectory=True,
            )
            logits = outputs["logits"]
            token_loss = self.loss_fn(logits, labels, ignore_index=IGNORE_LABEL_ID)
            ce_steps.append(token_loss.sum(-1) / loss_divisor)
            nextlat_steps.append(outputs["nextlat_loss"].to(torch.float32))
            residual_steps.append(self._task_residual_vec(logits, batch).to(torch.float32))
            ponder_steps.append(outputs["ponder_prob"].to(torch.float32))
            halt_steps.append(outputs["halt_prob"].to(torch.float32))
            mass_steps.append(outputs["ponder_mass"].to(torch.float32))
            state_delta_steps.append(outputs["state_delta"].to(torch.float32))
            logit_delta_steps.append(outputs["logit_delta"].to(torch.float32))
            if "solution_score_logits" in outputs:
                solution_score_steps.append(outputs["solution_score_logits"].to(torch.float32))
            if "chart_margin" in outputs:
                chart_margin_steps.append(outputs["chart_margin"].to(torch.float32))
            if "chart_gate" in outputs:
                chart_gate_steps.append(outputs["chart_gate"].to(torch.float32))

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                is_correct = mask & preds.eq(labels)
                cell_accuracy_steps.append(is_correct.to(torch.float32).sum(-1) / loss_divisor)
                exact_accuracy_steps.append((is_correct.sum(-1) == loss_counts) & valid_examples)
                prediction_steps.append(preds)
            final_outputs = outputs

        if not bool(carry.halted.all()):
            raise RuntimeError("clean_full trajectory did not terminate at the configured loop count.")

        ce = torch.stack(ce_steps, dim=0)
        nextlat = torch.stack(nextlat_steps, dim=0)
        residual = torch.stack(residual_steps, dim=0)
        ponder_prob = torch.stack(ponder_steps, dim=0)
        halt_prob = torch.stack(halt_steps, dim=0)
        cumulative_mass = torch.stack(mass_steps, dim=0)
        state_delta = torch.stack(state_delta_steps, dim=0)
        logit_delta = torch.stack(logit_delta_steps, dim=0)
        cell_accuracy = torch.stack(cell_accuracy_steps, dim=0)
        exact_accuracy = torch.stack(exact_accuracy_steps, dim=0)
        predictions = torch.stack(prediction_steps, dim=0)

        prior = self._trajectory_prior(device=ce.device)
        valid_prior = prior > 0
        log_prior = prior.clamp_min(1e-12).log().unsqueeze(1)
        kl_terms = torch.where(
            valid_prior.unsqueeze(1),
            ponder_prob * (ponder_prob.clamp_min(1e-12).log() - log_prior),
            torch.zeros_like(ponder_prob),
        )
        ponder_kl_per_example = kl_terms.sum(dim=0)
        ponder_kl = ponder_kl_per_example.sum()
        ponder_entropy = -torch.where(
            ponder_prob > 0,
            ponder_prob * ponder_prob.clamp_min(1e-12).log(),
            torch.zeros_like(ponder_prob),
        ).sum()

        step_numbers = torch.arange(1, self.config.loops + 1, dtype=torch.float32, device=ce.device).unsqueeze(1)
        expected_steps = (ponder_prob * step_numbers).sum(dim=0)
        ponder_mass = ponder_prob.sum(dim=0)
        ponder_task_loss = (ponder_prob * ce).sum()
        final_task_loss = ce[-1].sum()
        task_loss = (
            float(self.config.ponder_ce_weight) * ponder_task_loss
            + float(self.config.final_ce_weight) * final_task_loss
        )

        # NextLat regularizes every recurrent transition. It is not survival-
        # weighted, so the halt head cannot reduce this loss by stopping early.
        nextlat_loss = nextlat.mean(dim=0).sum()
        nextlat_term, nextlat_scale = self._normalize_aux(
            nextlat_loss,
            final_task_loss,
            bool(self.config.nextlat_normalize_to_ce),
        )
        task_residual_loss = (ponder_prob * residual).sum()
        task_residual_term, task_residual_scale = self._normalize_aux(
            task_residual_loss,
            final_task_loss,
            bool(self.config.task_residual_normalize_to_ce),
        )

        solution_score_loss = torch.zeros((), dtype=torch.float32, device=ce.device)
        solution_score_scale = torch.ones((), dtype=torch.float32, device=ce.device)
        solution_scores = None
        if solution_score_steps:
            solution_scores = torch.stack(solution_score_steps, dim=0)
            raw_solution_score_loss = F.binary_cross_entropy_with_logits(
                solution_scores,
                cell_accuracy.detach(),
                reduction="none",
            ).mean(dim=0).sum()
            solution_score_loss, solution_score_scale = self._normalize_aux(
                raw_solution_score_loss,
                final_task_loss,
                bool(self.config.solution_score_normalize_to_ce),
            )

        total_loss = (
            task_loss
            + float(self.config.nextlat_weight) * nextlat_term
            + float(self.config.task_residual_weight) * task_residual_term
            + float(self.config.ponder_kl_weight) * ponder_kl
            + float(self.config.ponder_step_cost) * expected_steps.sum()
            - float(self.config.ponder_entropy_weight) * ponder_entropy
            + float(self.config.solution_score_weight) * solution_score_loss
        )

        policy_indices = self._trajectory_policy_indices(halt_prob.detach(), cumulative_mass.detach())
        batch_indices = torch.arange(policy_indices.shape[0], device=policy_indices.device)
        policy_steps = policy_indices.to(torch.float32) + 1.0
        adaptive_cell_accuracy = cell_accuracy[policy_indices, batch_indices]
        adaptive_exact_accuracy = exact_accuracy[policy_indices, batch_indices]
        adaptive_predictions = predictions[policy_indices, batch_indices]

        exact_any = exact_accuracy.any(dim=0)
        oracle_hits = exact_accuracy.clone()
        oracle_hits[-1] |= ~exact_any
        oracle_indices = oracle_hits.to(torch.int64).argmax(dim=0)
        over_halt = policy_indices < oracle_indices
        under_halt = exact_any & (policy_indices > oracle_indices)

        decision_steps = torch.arange(1, self.config.loops + 1, device=ce.device)
        decision_mask = decision_steps.ge(int(self.config.ponder_min_steps)) & (
            decision_steps.remainder(max(1, int(self.config.readout_every))).eq(0)
        ) & decision_steps.lt(int(self.config.loops))
        if bool(decision_mask.any()):
            halt_oracle_brier = (
                halt_prob[decision_mask] - exact_accuracy[decision_mask].to(torch.float32)
            ).pow(2).mean(dim=0).sum()
        else:
            halt_oracle_brier = torch.zeros((), dtype=torch.float32, device=ce.device)

        solution_score_brier = torch.zeros((), dtype=torch.float32, device=ce.device)
        if solution_scores is not None:
            solution_score_brier = (
                torch.sigmoid(solution_scores[-1]) - cell_accuracy[-1]
            ).pow(2).sum()

        metrics: Dict[str, torch.Tensor] = {
            "count": count,
            "accuracy": cell_accuracy[-1].sum(),
            "exact_accuracy": exact_accuracy[-1].to(torch.float32).sum(),
            "adaptive_accuracy": adaptive_cell_accuracy.sum(),
            "adaptive_exact_accuracy": adaptive_exact_accuracy.to(torch.float32).sum(),
            "accuracy_per_compute": (adaptive_cell_accuracy / policy_steps.clamp_min(1.0)).sum(),
            "steps": policy_steps.sum(),
            "full_steps": count * float(self.config.loops),
            "ponder_expected_steps": expected_steps.detach().sum(),
            "ponder_mass": ponder_mass.detach().sum(),
            "ponder_mass_error": (ponder_mass.detach() - 1.0).abs().sum(),
            "ponder_prior_mass": count * prior.sum().detach(),
            "ponder_prob": ponder_prob[-1].detach().sum(),
            "ponder_alive": count * 0.0,
            "halt_prob": halt_prob[-1].detach().sum(),
            "over_halt_rate": over_halt.to(torch.float32).sum(),
            "under_halt_rate": under_halt.to(torch.float32).sum(),
            "halt_brier": halt_oracle_brier.detach(),
            "solution_score_brier": solution_score_brier.detach(),
            "lm_loss": task_loss.detach(),
            "ponder_ce_loss": ponder_task_loss.detach(),
            "final_ce_loss": final_task_loss.detach(),
            "nextlat_loss": nextlat_loss.detach(),
            "nextlat_scale": nextlat_scale.detach() * count,
            "task_residual_loss": task_residual_loss.detach(),
            "task_residual_scale": task_residual_scale.detach() * count,
            "ponder_kl_loss": ponder_kl.detach(),
            "ponder_step_loss": expected_steps.detach().sum(),
            "ponder_entropy": ponder_entropy.detach(),
            "halt_correctness_loss": torch.zeros((), dtype=torch.float32, device=ce.device),
            "solution_score_loss": solution_score_loss.detach(),
            "solution_score_scale": solution_score_scale.detach() * count,
            "state_delta": state_delta[-1].detach().sum(),
            "logit_delta": logit_delta[-1].detach().sum(),
        }

        log_step_metrics = (
            self.training and self.config.log_train_step_metrics
        ) or (
            (not self.training) and self.config.log_eval_step_metrics
        )
        if log_step_metrics:
            for step_index in range(self.config.loops):
                suffix = f"step_{step_index + 1:02d}"
                metrics[f"accuracy_{suffix}"] = cell_accuracy[step_index].sum()
                metrics[f"exact_accuracy_{suffix}"] = exact_accuracy[step_index].to(torch.float32).sum()
                metrics[f"halt_lambda_{suffix}"] = halt_prob[step_index].detach().sum()
                metrics[f"ponder_prob_{suffix}"] = ponder_prob[step_index].detach().sum()
                metrics[f"ponder_mass_{suffix}"] = cumulative_mass[step_index].detach().sum()
                metrics[f"nextlat_{suffix}"] = nextlat[step_index].detach().sum()
                metrics[f"state_delta_{suffix}"] = state_delta[step_index].detach().sum()
                metrics[f"logit_delta_{suffix}"] = logit_delta[step_index].detach().sum()
                if chart_margin_steps:
                    metrics[f"chart_margin_{suffix}"] = chart_margin_steps[step_index].detach().sum()
                if chart_gate_steps:
                    metrics[f"chart_gate_{suffix}"] = chart_gate_steps[step_index].detach().sum()

        final_outputs["preds"] = predictions[-1]
        final_outputs["adaptive_preds"] = adaptive_predictions
        returned_outputs: Dict[str, torch.Tensor] = {}
        if return_raw_outputs:
            returned_outputs["raw_outputs"] = final_outputs
        for key in return_keys:
            if key in final_outputs:
                returned_outputs[key] = final_outputs[key].detach()

        return carry, total_loss, metrics, returned_outputs, carry.halted.all()

    def _forward_step(
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
        ponder_task_loss = (ponder_prob * per_example_task_loss).sum()
        final_task_loss = (
            outputs["is_last_step"].to(torch.float32) * per_example_task_loss
        ).sum()
        ponder_ce_weight = float(getattr(self.config, "ponder_ce_weight", 1.0))
        final_ce_weight = float(getattr(self.config, "final_ce_weight", 0.0))
        task_loss = ponder_ce_weight * ponder_task_loss + final_ce_weight * final_task_loss

        nextlat_vec = outputs["nextlat_loss"].to(torch.float32)
        nextlat_loss = (alive_before * nextlat_vec).sum()
        nextlat_term, nextlat_scale = self._normalize_aux(
            nextlat_loss,
            task_loss,
            bool(getattr(self.config, "nextlat_normalize_to_ce", False)),
        )

        task_residual_vec = self._task_residual_vec(logits, new_carry.current_data).to(torch.float32)
        task_residual_loss = (alive_before * task_residual_vec).sum()
        task_residual_term, task_residual_scale = self._normalize_aux(
            task_residual_loss,
            task_loss,
            bool(getattr(self.config, "task_residual_normalize_to_ce", True)),
        )

        prior = self._prior_prob(new_carry.steps, outputs["is_last_step"]).to(torch.float32)
        ponder_kl = torch.where(
            ponder_prob > 0,
            ponder_prob * (ponder_prob.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log()),
            torch.zeros_like(ponder_prob),
        ).sum()
        ponder_entropy = -torch.where(
            ponder_prob > 0,
            ponder_prob * ponder_prob.clamp_min(1e-8).log(),
            torch.zeros_like(ponder_prob),
        ).sum()
        expected_step_loss = (ponder_prob * new_carry.steps.to(torch.float32)).sum()

        nextlat_weight = getattr(self.config, "nextlat_weight", 0.0)
        ponder_kl_weight = getattr(self.config, "ponder_kl_weight", 0.0)
        ponder_step_cost = getattr(self.config, "ponder_step_cost", 0.0)
        ponder_entropy_weight = getattr(self.config, "ponder_entropy_weight", 0.0)
        halt_correctness_weight = getattr(self.config, "halt_correctness_weight", 0.0)
        task_residual_weight = getattr(self.config, "task_residual_weight", 0.0)
        if halt_correctness_weight:
            halt_correctness_loss = F.binary_cross_entropy_with_logits(
                outputs["halt_logit"].to(torch.float32),
                seq_is_correct.to(torch.float32),
                reduction="none",
            )
            halt_correctness_loss = (alive_before * halt_correctness_loss).sum()
        else:
            halt_correctness_loss = torch.zeros((), dtype=torch.float32, device=logits.device)

        total_loss = (
            task_loss
            + nextlat_weight * nextlat_term
            + task_residual_weight * task_residual_term
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
            "ponder_ce_loss": ponder_task_loss.detach(),
            "final_ce_loss": final_task_loss.detach(),
            "nextlat_loss": nextlat_loss.detach(),
            "nextlat_scale": nextlat_scale.detach() * count,
            "task_residual_loss": task_residual_loss.detach(),
            "task_residual_scale": task_residual_scale.detach() * count,
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

        metrics = {name: value.detach() for name, value in metrics.items()}
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

    def forward(
        self,
        return_keys: Set[str],
        return_raw_outputs: bool = False,
        run_trajectory: bool = False,
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        if run_trajectory:
            return self._forward_trajectory(
                return_keys=return_keys,
                return_raw_outputs=return_raw_outputs,
                **model_kwargs,
            )
        return self._forward_step(
            return_keys=return_keys,
            return_raw_outputs=return_raw_outputs,
            **model_kwargs,
        )


# Backward-compatible aliases for older import strings while the new config uses TAPR.
TARMCarry = TAPRCarry
TARMConfig = TAPRConfig
TARMBlock = TAPRBlock
TARM_Inner = TAPRInner
TARM = TAPR
