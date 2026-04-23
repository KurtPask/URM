from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from models.common import trunc_normal_init_
from models.layers import CastedEmbedding, CastedLinear, CosSin, RotaryEmbedding, rms_norm
from models.pdat_tarm.layers import (
    AsynchronousScheduler,
    DualWorkspaceState,
    FactorSeedPool,
    PDATReasonerBlock,
    SchedulerState,
    SparseEntityFactorRouter,
    build_valid_mask,
    infer_task_type,
)
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class PDATTARMCarry:
    workspace: DualWorkspaceState
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class PDATTARMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float = 4.0
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int = 6
    L_cycles: int = 2
    H_cycles: int = 1
    task_type: str = "auto"
    num_factors: int = 32
    factor_seed_topk: int = 16
    entity_factor_topk: int = 8
    factor_entity_topk: int = 16
    entity_keep_ratio: float = 1.0
    factor_keep_ratio: float = 1.0
    support_tau_start: float = 1.0
    support_tau_end: float = 0.1
    conflict_tau_start: float = 1.0
    conflict_tau_end: float = 0.1
    tau_anneal_steps: int = 20000
    tau_schedule: str = "cosine"
    valuation_map: bool = True
    structural_factor_init: str = "weak"
    structural_prior_strength: float = 0.25
    attn_scale: float = 0.9
    support_scale: float = 0.5
    conflict_scale: float = 0.5
    mlp_scale: float = 0.7
    input_inject_scale: float = 0.25
    alpha_max: float = 0.60
    entity_alpha_init: float = 0.35
    factor_alpha_init: float = 0.35
    field_alpha_init: float = 0.50
    factor_refresh_alpha: float = 0.50
    entity_freeze_delta_threshold: float = 0.003
    entity_freeze_confidence_threshold: float = 0.985
    factor_freeze_delta_threshold: float = 0.003
    min_loops_before_entity_freeze: int = 1
    min_loops_before_factor_freeze: int = 2
    freeze_prefix_tokens: bool = True
    halt_delta_threshold: float = 0.010
    halt_unresolved_threshold: float = 0.30
    min_loops_before_halt: int = 2
    certificate_weight: float = 0.10
    forward_dtype: str = "bfloat16"


class PDATTARMInner(nn.Module):
    def __init__(self, config: PDATTARMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        self.pos_encoding_mode = self._resolve_pos_encoding_mode(self.config.pos_encodings)
        self.task_type = infer_task_type(self.config.task_type, self.config.seq_len, self.config.vocab_size)

        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.support_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=True)
        self.conflict_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

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

        self.factor_pool = FactorSeedPool(
            hidden_size=self.config.hidden_size,
            num_factors=self.config.num_factors,
            seed_topk=self.config.factor_seed_topk,
            valuation_map=self.config.valuation_map,
            norm_eps=self.config.rms_norm_eps,
            task_type=self.task_type,
            total_seq_len=self.config.seq_len + self.puzzle_emb_len,
            puzzle_emb_len=self.puzzle_emb_len,
            structural_factor_init=self.config.structural_factor_init,
            structural_prior_strength=self.config.structural_prior_strength,
        )
        self.router = SparseEntityFactorRouter(
            hidden_size=self.config.hidden_size,
            entity_topk=self.config.entity_factor_topk,
            factor_topk=self.config.factor_entity_topk,
            valuation_map=self.config.valuation_map,
            norm_eps=self.config.rms_norm_eps,
        )
        self.scheduler = AsynchronousScheduler(
            hidden_size=self.config.hidden_size,
            entity_keep_ratio=self.config.entity_keep_ratio,
            factor_keep_ratio=self.config.factor_keep_ratio,
            norm_eps=self.config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            [
                PDATReasonerBlock(
                    hidden_size=self.config.hidden_size,
                    expansion=self.config.expansion,
                    num_heads=self.config.num_heads,
                    norm_eps=self.config.rms_norm_eps,
                    valuation_map=self.config.valuation_map,
                    attn_scale=self.config.attn_scale,
                    support_scale=self.config.support_scale,
                    conflict_scale=self.config.conflict_scale,
                    mlp_scale=self.config.mlp_scale,
                    input_inject_scale=self.config.input_inject_scale,
                    alpha_max=self.config.alpha_max,
                    entity_alpha_init=self.config.entity_alpha_init,
                    factor_alpha_init=self.config.factor_alpha_init,
                    field_alpha_init=self.config.field_alpha_init,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.factor_refresh_logit = nn.Parameter(self._init_logit(self.config.factor_refresh_alpha))
        self.register_buffer(
            "init_entity_hidden",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )
        self.register_buffer(
            "init_factor_hidden",
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1.0),
            persistent=True,
        )
        self.register_buffer("tau_forward_step", torch.zeros((), dtype=torch.long), persistent=False)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    @staticmethod
    def _resolve_pos_encoding_mode(pos_encodings: str) -> str:
        aliases = {"learnable": "learned"}
        mode = aliases.get(pos_encodings, pos_encodings)
        if mode not in {"learned", "rope", "none"}:
            raise ValueError("pos_encodings must be one of: 'learned', 'learnable', 'rope', 'none'.")
        return mode

    def _init_logit(self, init_value: float) -> torch.Tensor:
        ratio = min(max(init_value / max(self.config.alpha_max, 1e-6), 1e-4), 1 - 1e-4)
        return torch.full((1, 1, 1), math.log(ratio / (1 - ratio)))

    def _seq_info(self) -> Dict[str, Optional[CosSin]]:
        if self.pos_encoding_mode == "rope":
            return {"cos_sin": self.rotary_emb()}
        return {"cos_sin": None}

    def _current_taus(self) -> Tuple[float, float]:
        step = int(self.tau_forward_step.item())
        frac = min(step / max(self.config.tau_anneal_steps, 1), 1.0)
        if self.config.tau_schedule == "cosine":
            frac = 0.5 - 0.5 * math.cos(math.pi * frac)
        support_tau = self.config.support_tau_start + frac * (self.config.support_tau_end - self.config.support_tau_start)
        conflict_tau = self.config.conflict_tau_start + frac * (self.config.conflict_tau_end - self.config.conflict_tau_start)
        return float(support_tau), float(conflict_tau)

    def _input_embeddings(self, inputs: torch.Tensor, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
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

    def empty_workspace(self, batch_size: int, device: torch.device) -> DualWorkspaceState:
        total_seq_len = self.config.seq_len + self.puzzle_emb_len
        factor_shape = (batch_size, self.config.num_factors, self.config.hidden_size)
        entity_shape = (batch_size, total_seq_len, self.config.hidden_size)

        entity_init = self.init_entity_hidden.to(device).view(1, 1, -1).expand(entity_shape)
        factor_init = self.init_factor_hidden.to(device).view(1, 1, -1).expand(factor_shape)
        zeros_entity = torch.zeros(entity_shape, dtype=self.forward_dtype, device=device)
        zeros_factor = torch.zeros(factor_shape, dtype=self.forward_dtype, device=device)
        entity_frozen = torch.zeros((batch_size, total_seq_len), dtype=torch.bool, device=device)
        if self.config.freeze_prefix_tokens and self.puzzle_emb_len > 0:
            entity_frozen[:, : self.puzzle_emb_len] = True
        factor_frozen = torch.zeros((batch_size, self.config.num_factors), dtype=torch.bool, device=device)

        return DualWorkspaceState(
            entity_hidden=entity_init.clone(),
            factor_hidden=factor_init.clone(),
            entity_support=zeros_entity,
            entity_conflict=zeros_entity.clone(),
            factor_support=zeros_factor,
            factor_conflict=zeros_factor.clone(),
            entity_frozen=entity_frozen,
            factor_frozen=factor_frozen,
        )

    def reset_workspace(self, reset_flag: torch.Tensor, workspace: DualWorkspaceState) -> DualWorkspaceState:
        device = workspace.entity_hidden.device
        reset = reset_flag.to(device).view(-1, 1, 1)

        init_workspace = self.empty_workspace(reset_flag.shape[0], device)

        return DualWorkspaceState(
            entity_hidden=torch.where(reset, init_workspace.entity_hidden, workspace.entity_hidden),
            factor_hidden=torch.where(reset, init_workspace.factor_hidden, workspace.factor_hidden),
            entity_support=torch.where(reset, init_workspace.entity_support, workspace.entity_support),
            entity_conflict=torch.where(reset, init_workspace.entity_conflict, workspace.entity_conflict),
            factor_support=torch.where(reset, init_workspace.factor_support, workspace.factor_support),
            factor_conflict=torch.where(reset, init_workspace.factor_conflict, workspace.factor_conflict),
            entity_frozen=torch.where(reset.view(-1, 1), init_workspace.entity_frozen, workspace.entity_frozen),
            factor_frozen=torch.where(reset.view(-1, 1), init_workspace.factor_frozen, workspace.factor_frozen),
        )

    @staticmethod
    def _detach_workspace(workspace: DualWorkspaceState) -> DualWorkspaceState:
        return DualWorkspaceState(
            entity_hidden=workspace.entity_hidden.detach(),
            factor_hidden=workspace.factor_hidden.detach(),
            entity_support=workspace.entity_support.detach(),
            entity_conflict=workspace.entity_conflict.detach(),
            factor_support=workspace.factor_support.detach(),
            factor_conflict=workspace.factor_conflict.detach(),
            entity_frozen=workspace.entity_frozen.detach(),
            factor_frozen=workspace.factor_frozen.detach(),
        )

    @staticmethod
    def _select_workspace(previous: DualWorkspaceState, updated: DualWorkspaceState, freeze_mask: torch.Tensor) -> DualWorkspaceState:
        mask = freeze_mask.view(-1, 1, 1)
        return DualWorkspaceState(
            entity_hidden=torch.where(mask, previous.entity_hidden, updated.entity_hidden),
            factor_hidden=torch.where(mask, previous.factor_hidden, updated.factor_hidden),
            entity_support=torch.where(mask, previous.entity_support, updated.entity_support),
            entity_conflict=torch.where(mask, previous.entity_conflict, updated.entity_conflict),
            factor_support=torch.where(mask, previous.factor_support, updated.factor_support),
            factor_conflict=torch.where(mask, previous.factor_conflict, updated.factor_conflict),
            entity_frozen=torch.where(mask.squeeze(-1), previous.entity_frozen, updated.entity_frozen),
            factor_frozen=torch.where(mask.squeeze(-1), previous.factor_frozen, updated.factor_frozen),
        )

    def _full_valid_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        token_mask = build_valid_mask(inputs, self.task_type)
        if self.puzzle_emb_len <= 0:
            return token_mask
        prefix = torch.ones(
            inputs.shape[0],
            self.puzzle_emb_len,
            dtype=torch.bool,
            device=inputs.device,
        )
        return torch.cat((prefix, token_mask), dim=1)

    def _macro_step(
        self,
        workspace: DualWorkspaceState,
        input_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
        cos_sin: Optional[CosSin],
        support_tau: float,
        conflict_tau: float,
    ) -> Tuple[DualWorkspaceState, SchedulerState]:
        factor_seed = self.factor_pool(workspace.entity_hidden, valid_mask)
        factor_alpha = self.config.alpha_max * torch.sigmoid(self.factor_refresh_logit).to(workspace.entity_hidden.dtype)
        factor_gate = (~workspace.factor_frozen).unsqueeze(-1).to(workspace.entity_hidden.dtype)
        factor_hidden = workspace.factor_hidden + factor_alpha * factor_gate * (factor_seed - workspace.factor_hidden)
        workspace = replace(workspace, factor_hidden=factor_hidden)

        routing = self.router(workspace.entity_hidden, workspace.factor_hidden, valid_mask)
        scheduler = self.scheduler(
            entity_hidden=workspace.entity_hidden,
            entity_support=workspace.entity_support,
            entity_conflict=workspace.entity_conflict,
            factor_hidden=workspace.factor_hidden,
            factor_support=workspace.factor_support,
            factor_conflict=workspace.factor_conflict,
            entity_valid_mask=routing.entity_valid_mask,
            factor_valid_mask=routing.factor_valid_mask,
            entity_frozen=workspace.entity_frozen,
            factor_frozen=workspace.factor_frozen,
        )

        for _ in range(self.config.L_cycles):
            for layer in self.layers:
                workspace = layer(
                    workspace=workspace,
                    routing=routing,
                    scheduler=scheduler,
                    input_embeddings=input_embeddings,
                    cos_sin=cos_sin,
                    support_tau=support_tau,
                    conflict_tau=conflict_tau,
                )
        return workspace, scheduler

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        denom = mask_f.sum(dim=-1).clamp(min=1.0)
        return (values * mask_f).sum(dim=-1) / denom

    def _certificate_aux_loss(
        self,
        support_logits: torch.Tensor,
        conflict_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = labels.ne(-100)
        if not valid_mask.any():
            return support_logits.new_zeros(())

        masked_labels = torch.where(valid_mask, labels, 0).to(torch.long)

        support_loss = F.cross_entropy(
            support_logits[valid_mask].to(torch.float32),
            masked_labels[valid_mask],
        )

        correct_conflict = torch.gather(
            conflict_logits.to(torch.float32),
            dim=-1,
            index=masked_labels.unsqueeze(-1),
        ).squeeze(-1)

        incorrect_conflict = conflict_logits.to(torch.float32).masked_fill(
            F.one_hot(masked_labels, num_classes=conflict_logits.shape[-1]).bool(),
            float("-inf"),
        ).amax(dim=-1)

        violation_loss = F.softplus(1.0 + correct_conflict - incorrect_conflict)
        violation_loss = violation_loss[valid_mask].mean()
        return self.config.certificate_weight * (support_loss + violation_loss)

    def forward(
        self,
        workspace: DualWorkspaceState,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[DualWorkspaceState, Dict[str, torch.Tensor]]:
        del compute_target_q

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        valid_mask = self._full_valid_mask(batch["inputs"])
        seq_info = self._seq_info()
        if self.training:
            self.tau_forward_step.add_(1)
        support_tau, conflict_tau = self._current_taus()

        steps_used = torch.full(
            (input_embeddings.shape[0],),
            self.config.loops,
            dtype=torch.int32,
            device=input_embeddings.device,
        )
        halted = torch.zeros_like(steps_used, dtype=torch.bool)
        router_metrics = {}
        delta = torch.zeros(input_embeddings.shape[0], dtype=torch.float32, device=input_embeddings.device)
        q_logits = torch.zeros(input_embeddings.shape[0], 2, dtype=torch.float32, device=input_embeddings.device)
        mean_token_confidence = torch.zeros((), dtype=torch.float32, device=input_embeddings.device)
        mean_token_delta = torch.zeros((), dtype=torch.float32, device=input_embeddings.device)
        frozen_token_ratio = torch.zeros((), dtype=torch.float32, device=input_embeddings.device)
        token_valid_mask = valid_mask[:, self.puzzle_emb_len :]

        for loop_idx in range(self.config.loops):
            previous = workspace
            updated, scheduler = self._macro_step(
                workspace=workspace,
                input_embeddings=input_embeddings,
                valid_mask=valid_mask,
                support_tau=support_tau,
                conflict_tau=conflict_tau,
                **seq_info,
            )

            token_delta = (
                (updated.entity_hidden[:, self.puzzle_emb_len :] - previous.entity_hidden[:, self.puzzle_emb_len :])
                .abs()
                .mean(dim=-1)
                .to(torch.float32)
            )
            delta = self._masked_mean(token_delta, token_valid_mask)
            q_logits = self.q_head(rms_norm(updated.entity_hidden[:, 0], variance_epsilon=self.config.rms_norm_eps)).to(torch.float32)
            q_halt_logits = q_logits[:, 0]
            q_continue_logits = q_logits[:, 1]

            support_logits = self.support_head(rms_norm(updated.entity_hidden, variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len :]
            conflict_logits = self.conflict_head(rms_norm(updated.entity_hidden, variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len :]
            token_confidence = torch.softmax(support_logits.to(torch.float32), dim=-1).amax(dim=-1)
            unresolved = 1.0 - self._masked_mean(token_confidence, token_valid_mask)

            next_entity_frozen = updated.entity_frozen.clone()
            next_factor_frozen = updated.factor_frozen.clone()
            if self.config.freeze_prefix_tokens and self.puzzle_emb_len > 0:
                next_entity_frozen[:, : self.puzzle_emb_len] = True

            if loop_idx + 1 >= self.config.min_loops_before_entity_freeze:
                entity_can_freeze = (
                    token_valid_mask
                    & (token_delta <= self.config.entity_freeze_delta_threshold)
                    & (token_confidence >= self.config.entity_freeze_confidence_threshold)
                )
                next_entity_frozen[:, self.puzzle_emb_len :] = (
                    next_entity_frozen[:, self.puzzle_emb_len :] | entity_can_freeze
                )

            factor_delta = (
                (updated.factor_hidden - previous.factor_hidden).abs().mean(dim=-1).to(torch.float32)
            )
            if loop_idx + 1 >= self.config.min_loops_before_factor_freeze:
                factor_can_freeze = factor_delta <= self.config.factor_freeze_delta_threshold
                next_factor_frozen = next_factor_frozen | factor_can_freeze

            updated = replace(updated, entity_frozen=next_entity_frozen, factor_frozen=next_factor_frozen)
            all_tokens_frozen = ((~token_valid_mask) | updated.entity_frozen[:, self.puzzle_emb_len :]).all(dim=-1)
            mean_token_confidence = self._masked_mean(token_confidence, token_valid_mask).mean()
            mean_token_delta = delta.mean()
            frozen_token_ratio = (
                (
                    (updated.entity_frozen[:, self.puzzle_emb_len :] & token_valid_mask)
                    .to(torch.float32)
                    .sum()
                )
                / token_valid_mask.to(torch.float32).sum().clamp(min=1.0)
            )

            can_halt = torch.zeros_like(halted)
            if loop_idx + 1 >= self.config.min_loops_before_halt:
                can_halt = all_tokens_frozen | (
                    (delta <= self.config.halt_delta_threshold)
                    & ((unresolved <= self.config.halt_unresolved_threshold) | (q_halt_logits > q_continue_logits))
                )
            new_halted = can_halt & (~halted)
            steps_used = torch.where(
                new_halted,
                torch.full_like(steps_used, loop_idx + 1),
                steps_used,
            )
            workspace = self._select_workspace(previous, updated, halted)
            halted = halted | can_halt
            router_metrics = scheduler.metrics
            if bool(halted.all()):
                break

        entity_hidden = workspace.entity_hidden
        logits = self.lm_head(rms_norm(entity_hidden, variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len :]
        support_logits = self.support_head(rms_norm(entity_hidden, variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len :]
        conflict_logits = self.conflict_head(rms_norm(entity_hidden, variance_epsilon=self.config.rms_norm_eps))[:, self.puzzle_emb_len :]
        aux_loss = self._certificate_aux_loss(support_logits, conflict_logits, batch["labels"])

        outputs = {
            "logits": logits,
            "q_halt_logits": q_logits[:, 0],
            "q_continue_logits": q_logits[:, 1],
            "support_logits": support_logits,
            "conflict_logits": conflict_logits,
            "steps_used": steps_used,
            "moe_aux_loss": aux_loss,
            "router_metrics": {
                **router_metrics,
                "state_delta": delta.mean(),
                "mean_steps": steps_used.to(torch.float32).mean(),
                "support_tau": torch.tensor(support_tau, dtype=torch.float32, device=input_embeddings.device),
                "conflict_tau": torch.tensor(conflict_tau, dtype=torch.float32, device=input_embeddings.device),
                "mean_token_confidence": mean_token_confidence,
                "mean_token_delta": mean_token_delta,
                "frozen_token_ratio": frozen_token_ratio,
            },
        }

        return self._detach_workspace(workspace), outputs


class PDATTARM(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = PDATTARMConfig(**config_dict)
        self.inner = PDATTARMInner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> PDATTARMCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        workspace = self.inner.empty_workspace(batch_size, device)
        return PDATTARMCarry(
            workspace=workspace,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: PDATTARMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q: bool = False,
    ) -> Tuple[PDATTARMCarry, Dict[str, torch.Tensor]]:
        reset_workspace = self.inner.reset_workspace(carry.halted, carry.workspace)
        current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        new_workspace, outputs = self.inner(reset_workspace, current_data, compute_target_q=compute_target_q)
        steps = outputs["router_metrics"]["mean_steps"].new_full(
            (batch["inputs"].shape[0],),
            self.config.loops,
        ).to(torch.int32)

        return (
            PDATTARMCarry(
                workspace=new_workspace,
                steps=outputs["steps_used"].to(torch.int32),
                halted=torch.ones_like(steps, dtype=torch.bool),
                current_data=current_data,
            ),
            outputs,
        )
