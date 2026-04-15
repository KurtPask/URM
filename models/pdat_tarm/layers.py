from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear,
    ConvSwiGLU,
    CosSin,
    TropicalAttentionV2,
    rms_norm,
)


@dataclass
class RoutingState:
    entity_to_factor_index: torch.Tensor
    entity_to_factor_score: torch.Tensor
    factor_to_entity_index: torch.Tensor
    factor_to_entity_score: torch.Tensor
    entity_valid_mask: torch.Tensor
    factor_valid_mask: torch.Tensor


@dataclass
class SchedulerState:
    entity_gate: torch.Tensor
    factor_gate: torch.Tensor
    metrics: Dict[str, torch.Tensor]


@dataclass
class DualWorkspaceState:
    entity_hidden: torch.Tensor
    factor_hidden: torch.Tensor
    entity_support: torch.Tensor
    entity_conflict: torch.Tensor
    factor_support: torch.Tensor
    factor_conflict: torch.Tensor
    entity_frozen: torch.Tensor
    factor_frozen: torch.Tensor


def infer_task_type(task_type: str, seq_len: int, vocab_size: int) -> str:
    if task_type != "auto":
        return task_type
    if seq_len == 81:
        return "sudoku"
    if seq_len >= 900 and vocab_size <= 16:
        return "arc"
    return "generic"


def build_valid_mask(inputs: torch.Tensor, task_type: str) -> torch.Tensor:
    if task_type == "arc":
        return inputs.ne(0)
    return torch.ones_like(inputs, dtype=torch.bool)


def build_factor_seed_prior(
    task_type: str,
    total_seq_len: int,
    puzzle_emb_len: int,
    num_factors: int,
) -> torch.Tensor:
    prior = torch.zeros(num_factors, total_seq_len, dtype=torch.float32)
    token_len = total_seq_len - puzzle_emb_len
    if token_len <= 0:
        return prior

    offset = puzzle_emb_len
    if task_type == "sudoku" and token_len == 81:
        side = 9
        for row in range(min(side, num_factors)):
            start = offset + row * side
            prior[row, start : start + side] = 1.0
        for col in range(min(side, max(num_factors - side, 0))):
            prior[side + col, offset + col : offset + token_len : side] = 1.0
        for box in range(min(side, max(num_factors - 2 * side, 0))):
            box_row = box // 3
            box_col = box % 3
            factor_idx = 2 * side + box
            for local_row in range(3):
                start = offset + (box_row * 3 + local_row) * side + box_col * 3
                prior[factor_idx, start : start + 3] = 1.0
        return prior

    side = int(math.isqrt(token_len))
    if side * side != token_len:
        return prior

    for row in range(min(side, num_factors)):
        start = offset + row * side
        prior[row, start : start + side] = 1.0
    for col in range(min(side, max(num_factors - side, 0))):
        prior[side + col, offset + col : offset + token_len : side] = 1.0
    return prior


def pairwise_hilbert_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    diff = query.unsqueeze(2) - key.unsqueeze(1)
    return -(diff.amax(dim=-1) - diff.amin(dim=-1))


def soft_tropical_max(values: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-4)
    return tau * torch.logsumexp(values / tau, dim=dim)


def soft_tropical_min(values: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    tau = max(float(tau), 1e-4)
    return -tau * torch.logsumexp(-values / tau, dim=dim)


def gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, _, hidden_size = values.shape
    target_count, topk = indices.shape[1], indices.shape[2]
    expanded = indices.unsqueeze(-1).expand(batch_size, target_count, topk, hidden_size)
    source = values.unsqueeze(1).expand(batch_size, target_count, values.shape[1], hidden_size)
    return torch.gather(source, dim=2, index=expanded)


def gated_lerp(current: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    step = alpha.to(current.dtype) * gate.to(current.dtype)
    return current + step * (proposal - current)


class FactorSeedPool(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_factors: int,
        seed_topk: int,
        valuation_map: bool,
        norm_eps: float,
        task_type: str,
        total_seq_len: int,
        puzzle_emb_len: int,
        structural_factor_init: str,
        structural_prior_strength: float,
    ) -> None:
        super().__init__()
        self.seed_topk = seed_topk
        self.valuation_map = valuation_map
        self.norm_eps = norm_eps
        self.structural_factor_init = structural_factor_init
        self.structural_prior_strength = structural_prior_strength

        self.factor_queries = nn.Parameter(trunc_normal_init_(torch.empty(num_factors, hidden_size), std=1.0 / math.sqrt(hidden_size)))
        self.factor_query_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.entity_key_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.entity_value_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.seed_proj = CastedLinear(hidden_size, hidden_size, bias=True)
        self.prior_gate_logit = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.register_buffer(
            "factor_seed_prior",
            build_factor_seed_prior(task_type, total_seq_len, puzzle_emb_len, num_factors),
            persistent=True,
        )

    def _valuation(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        if self.valuation_map:
            return torch.log1p(F.relu(tensor))
        return tensor

    def forward(self, entity_hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = entity_hidden.shape
        normed = rms_norm(entity_hidden, variance_epsilon=self.norm_eps)
        entity_keys = self._valuation(self.entity_key_proj(normed))
        entity_values = self.entity_value_proj(normed)

        factor_queries = self._valuation(
            self.factor_query_proj(self.factor_queries.to(normed.dtype)).unsqueeze(0).expand(batch_size, -1, -1)
        )
        scores = pairwise_hilbert_scores(factor_queries, entity_keys)
        if self.structural_factor_init != "none" and self.structural_prior_strength > 0:
            prior_strength = (
                torch.sigmoid(self.prior_gate_logit).to(scores.dtype) * float(self.structural_prior_strength)
            )
            scores = scores + prior_strength * self.factor_seed_prior[:, :seq_len].unsqueeze(0).to(scores.dtype)
        scores = torch.where(valid_mask.unsqueeze(1), scores, torch.full_like(scores, -1e9))

        topk = min(self.seed_topk, seq_len)
        factor_scores, factor_index = scores.topk(k=topk, dim=-1)
        gathered = gather_neighbors(entity_values, factor_index)
        pooled = soft_tropical_max(gathered + factor_scores.unsqueeze(-1), dim=2, tau=1.0)
        learned_seed = self.seed_proj(self.factor_queries.to(entity_hidden.dtype)).unsqueeze(0).expand_as(pooled)
        return learned_seed + pooled


class SparseEntityFactorRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        entity_topk: int,
        factor_topk: int,
        valuation_map: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.entity_topk = entity_topk
        self.factor_topk = factor_topk
        self.valuation_map = valuation_map
        self.norm_eps = norm_eps

        self.entity_query_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.factor_key_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def _valuation(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(torch.float32)
        if self.valuation_map:
            return torch.log1p(F.relu(tensor))
        return tensor

    def forward(self, entity_hidden: torch.Tensor, factor_hidden: torch.Tensor, entity_valid_mask: torch.Tensor) -> RoutingState:
        batch_size, seq_len, _ = entity_hidden.shape
        num_factors = factor_hidden.shape[1]

        entity_query = self._valuation(self.entity_query_proj(rms_norm(entity_hidden, variance_epsilon=self.norm_eps)))
        factor_key = self._valuation(self.factor_key_proj(rms_norm(factor_hidden, variance_epsilon=self.norm_eps)))

        scores = pairwise_hilbert_scores(entity_query, factor_key)
        scores = torch.where(entity_valid_mask.unsqueeze(-1), scores, torch.full_like(scores, -1e9))

        entity_topk = min(self.entity_topk, num_factors)
        factor_topk = min(self.factor_topk, seq_len)

        entity_scores, entity_index = scores.topk(k=entity_topk, dim=-1)
        factor_scores, factor_index = scores.transpose(1, 2).topk(k=factor_topk, dim=-1)

        return RoutingState(
            entity_to_factor_index=entity_index,
            entity_to_factor_score=entity_scores.to(entity_hidden.dtype),
            factor_to_entity_index=factor_index,
            factor_to_entity_score=factor_scores.to(entity_hidden.dtype),
            entity_valid_mask=entity_valid_mask,
            factor_valid_mask=torch.ones(
                batch_size,
                num_factors,
                dtype=torch.bool,
                device=entity_hidden.device,
            ),
        )


class AsynchronousScheduler(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        entity_keep_ratio: float,
        factor_keep_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.entity_keep_ratio = entity_keep_ratio
        self.factor_keep_ratio = factor_keep_ratio
        self.norm_eps = norm_eps

        self.entity_score = CastedLinear(hidden_size * 3, 1, bias=True)
        self.factor_score = CastedLinear(hidden_size * 3, 1, bias=True)

    @staticmethod
    def _topk_gate(scores: torch.Tensor, keep_ratio: float, valid_mask: torch.Tensor) -> torch.Tensor:
        if keep_ratio >= 1.0:
            return valid_mask.unsqueeze(-1).to(scores.dtype)

        count = scores.shape[1]
        keep = max(1, int(math.ceil(count * keep_ratio)))
        masked_scores = torch.where(valid_mask, scores, torch.full_like(scores, -1e9))
        topk_index = masked_scores.topk(k=min(keep, count), dim=-1).indices
        gate = torch.zeros_like(scores, dtype=torch.bool)
        gate.scatter_(1, topk_index, True)
        gate = gate & valid_mask
        return gate.unsqueeze(-1).to(scores.dtype)

    def forward(
        self,
        entity_hidden: torch.Tensor,
        entity_support: torch.Tensor,
        entity_conflict: torch.Tensor,
        factor_hidden: torch.Tensor,
        factor_support: torch.Tensor,
        factor_conflict: torch.Tensor,
        entity_valid_mask: torch.Tensor,
        factor_valid_mask: torch.Tensor,
        entity_frozen: torch.Tensor,
        factor_frozen: torch.Tensor,
    ) -> SchedulerState:
        entity_features = torch.cat(
            (
                rms_norm(entity_hidden, variance_epsilon=self.norm_eps),
                entity_support.abs(),
                entity_conflict.abs(),
            ),
            dim=-1,
        )
        factor_features = torch.cat(
            (
                rms_norm(factor_hidden, variance_epsilon=self.norm_eps),
                factor_support.abs(),
                factor_conflict.abs(),
            ),
            dim=-1,
        )

        entity_scores = self.entity_score(entity_features).squeeze(-1)
        factor_scores = self.factor_score(factor_features).squeeze(-1)

        entity_gate = self._topk_gate(entity_scores, self.entity_keep_ratio, entity_valid_mask)
        factor_gate = self._topk_gate(factor_scores, self.factor_keep_ratio, factor_valid_mask)
        entity_gate = entity_gate * (~entity_frozen).unsqueeze(-1).to(entity_gate.dtype)
        factor_gate = factor_gate * (~factor_frozen).unsqueeze(-1).to(factor_gate.dtype)

        metrics = {
            "entity_active_ratio": entity_gate.mean(),
            "factor_active_ratio": factor_gate.mean(),
        }
        return SchedulerState(entity_gate=entity_gate, factor_gate=factor_gate, metrics=metrics)


class PDATReasonerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        num_heads: int,
        norm_eps: float,
        valuation_map: bool,
        attn_scale: float,
        support_scale: float,
        conflict_scale: float,
        mlp_scale: float,
        input_inject_scale: float,
        alpha_max: float,
        entity_alpha_init: float,
        factor_alpha_init: float,
        field_alpha_init: float,
    ) -> None:
        super().__init__()
        self.norm_eps = norm_eps
        self.attn_scale = attn_scale
        self.support_scale = support_scale
        self.conflict_scale = conflict_scale
        self.mlp_scale = mlp_scale
        self.input_inject_scale = input_inject_scale
        self.alpha_max = alpha_max

        self.entity_attn = TropicalAttentionV2(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
            valuation_map=valuation_map,
        )
        self.entity_mlp = ConvSwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.factor_mlp = ConvSwiGLU(hidden_size=hidden_size, expansion=expansion)

        self.entity_support_source = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        self.entity_conflict_source = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        self.factor_support_source = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        self.factor_conflict_source = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        self.entity_update_proj = CastedLinear(hidden_size * 4, hidden_size, bias=True)
        self.factor_update_proj = CastedLinear(hidden_size * 3, hidden_size, bias=True)

        self.entity_alpha_logit = nn.Parameter(self._init_logit(entity_alpha_init, alpha_max))
        self.factor_alpha_logit = nn.Parameter(self._init_logit(factor_alpha_init, alpha_max))
        self.field_alpha_logit = nn.Parameter(self._init_logit(field_alpha_init, alpha_max))

    @staticmethod
    def _init_logit(init_value: float, alpha_max: float) -> torch.Tensor:
        ratio = min(max(init_value / max(alpha_max, 1e-6), 1e-4), 1 - 1e-4)
        return torch.full((1, 1, 1), math.log(ratio / (1 - ratio)))

    def _aggregate_support(
        self,
        source: torch.Tensor,
        index: torch.Tensor,
        score: torch.Tensor,
        support_tau: float,
    ) -> torch.Tensor:
        gathered = gather_neighbors(source, index)
        return soft_tropical_max(gathered + score.unsqueeze(-1), dim=2, tau=support_tau)

    def _aggregate_conflict(
        self,
        source: torch.Tensor,
        index: torch.Tensor,
        score: torch.Tensor,
        conflict_tau: float,
    ) -> torch.Tensor:
        gathered = gather_neighbors(source, index)
        return soft_tropical_min(gathered - score.unsqueeze(-1), dim=2, tau=conflict_tau)

    def forward(
        self,
        workspace: DualWorkspaceState,
        routing: RoutingState,
        scheduler: SchedulerState,
        input_embeddings: torch.Tensor,
        cos_sin: CosSin | None,
        support_tau: float,
        conflict_tau: float,
    ) -> DualWorkspaceState:
        entity_hidden = workspace.entity_hidden
        factor_hidden = workspace.factor_hidden

        entity_source_input = torch.cat(
            (
                rms_norm(entity_hidden, variance_epsilon=self.norm_eps),
                workspace.entity_support,
            ),
            dim=-1,
        )
        entity_conflict_input = torch.cat(
            (
                rms_norm(entity_hidden, variance_epsilon=self.norm_eps),
                workspace.entity_conflict,
            ),
            dim=-1,
        )
        entity_support_src = self.entity_support_source(entity_source_input)
        entity_conflict_src = self.entity_conflict_source(entity_conflict_input)

        factor_support_msg = self._aggregate_support(
            entity_support_src,
            routing.factor_to_entity_index,
            routing.factor_to_entity_score.to(entity_hidden.dtype),
            support_tau,
        )
        factor_conflict_msg = self._aggregate_conflict(
            entity_conflict_src,
            routing.factor_to_entity_index,
            routing.factor_to_entity_score.to(entity_hidden.dtype),
            conflict_tau,
        )

        factor_pre = torch.cat((factor_hidden, factor_support_msg, factor_conflict_msg), dim=-1)
        factor_mlp_input = factor_hidden + factor_support_msg - factor_conflict_msg
        factor_candidate = factor_hidden + self.factor_update_proj(factor_pre) + self.factor_mlp(
            rms_norm(factor_mlp_input, variance_epsilon=self.norm_eps)
        )

        factor_alpha = self.alpha_max * torch.sigmoid(self.factor_alpha_logit).to(entity_hidden.dtype)
        field_alpha = self.alpha_max * torch.sigmoid(self.field_alpha_logit).to(entity_hidden.dtype)

        factor_hidden = gated_lerp(factor_hidden, factor_candidate, factor_alpha, scheduler.factor_gate)
        factor_support = gated_lerp(workspace.factor_support, factor_support_msg, field_alpha, scheduler.factor_gate)
        factor_conflict = gated_lerp(workspace.factor_conflict, factor_conflict_msg, field_alpha, scheduler.factor_gate)

        factor_source_input = torch.cat(
            (
                rms_norm(factor_hidden, variance_epsilon=self.norm_eps),
                factor_support,
            ),
            dim=-1,
        )
        factor_conflict_input = torch.cat(
            (
                rms_norm(factor_hidden, variance_epsilon=self.norm_eps),
                factor_conflict,
            ),
            dim=-1,
        )
        factor_support_src = self.factor_support_source(factor_source_input)
        factor_conflict_src = self.factor_conflict_source(factor_conflict_input)

        entity_support_msg = self._aggregate_support(
            factor_support_src,
            routing.entity_to_factor_index,
            routing.entity_to_factor_score.to(entity_hidden.dtype),
            support_tau,
        )
        entity_conflict_msg = self._aggregate_conflict(
            factor_conflict_src,
            routing.entity_to_factor_index,
            routing.entity_to_factor_score.to(entity_hidden.dtype),
            conflict_tau,
        )

        attn_input = rms_norm(entity_hidden, variance_epsilon=self.norm_eps)
        attn_output = self.entity_attn(cos_sin=cos_sin, hidden_states=attn_input)

        entity_pre = torch.cat((entity_hidden, attn_output, entity_support_msg, entity_conflict_msg), dim=-1)
        entity_candidate = (
            entity_hidden
            + self.input_inject_scale * input_embeddings
            + self.attn_scale * attn_output
            + self.support_scale * entity_support_msg
            - self.conflict_scale * entity_conflict_msg
            + self.entity_update_proj(entity_pre)
        )
        entity_candidate = entity_candidate + self.mlp_scale * self.entity_mlp(
            rms_norm(entity_candidate, variance_epsilon=self.norm_eps)
        )

        entity_alpha = self.alpha_max * torch.sigmoid(self.entity_alpha_logit).to(entity_hidden.dtype)
        entity_hidden = gated_lerp(entity_hidden, entity_candidate, entity_alpha, scheduler.entity_gate)
        entity_support = gated_lerp(workspace.entity_support, entity_support_msg, field_alpha, scheduler.entity_gate)
        entity_conflict = gated_lerp(workspace.entity_conflict, entity_conflict_msg, field_alpha, scheduler.entity_gate)

        return DualWorkspaceState(
            entity_hidden=entity_hidden,
            factor_hidden=factor_hidden,
            entity_support=entity_support,
            entity_conflict=entity_conflict,
            factor_support=factor_support,
            factor_conflict=factor_conflict,
            entity_frozen=workspace.entity_frozen,
            factor_frozen=workspace.factor_frozen,
        )
