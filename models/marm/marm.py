from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm,
    ConvSwiGLU,
    RotaryEmbedding,
    CosSin,
    CastedEmbedding,
    CastedLinear,
    apply_rotary_pos_emb,
)
from models.sparse_embedding import CastedSparseEmbedding


_flash_attn_import_error: Optional[Exception] = None
flash_attn_func = None
try:
    from flash_attn_interface import flash_attn_func  # type: ignore[assignment]
except ImportError:
    try:
        from flash_attn import flash_attn_func  # type: ignore[assignment]
    except ImportError as exc:
        _flash_attn_import_error = exc

_FLASH_ATTN_ENABLED = flash_attn_func is not None


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    x: [bs, seq, kv_heads, head_dim]
    returns: [bs, seq, kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    bs, seq, kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(bs, seq, kv_heads, n_rep, head_dim)
    return x.reshape(bs, seq, kv_heads * n_rep, head_dim)


class MixedAttention(nn.Module):
    """
    Routes part of the heads through standard attention (FlashAttention when available,
    otherwise SDPA fallback) and the remainder through tropical attention.

    This version is self-contained so marm.py does not depend on models.layers
    having a MixedAttention implementation.
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False,
        attn_dropout: float = 0.0,
        valuation_map: bool = True,
        flash_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if head_dim != hidden_size // num_heads:
            raise ValueError("head_dim must equal hidden_size // num_heads.")
        if num_heads % num_key_value_heads != 0:
            raise ValueError("num_heads must be divisible by num_key_value_heads.")
        if not (0.0 < flash_fraction < 1.0):
            raise ValueError("flash_fraction must be strictly between 0 and 1.")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.output_size = num_heads * head_dim
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.valuation_map = valuation_map
        self.branch_norm_eps = 1e-5
        self.flash_gain = nn.Parameter(torch.tensor(1.0))
        self.tropical_gain = nn.Parameter(torch.tensor(1.0))
        
        self.q_per_kv = num_heads // num_key_value_heads

        self.qkv_proj = CastedLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

        # Split by whole KV groups so GQA/MQA alignment stays valid.
        if num_key_value_heads > 1:
            flash_kv_heads = int(round(num_key_value_heads * flash_fraction))
            flash_kv_heads = max(1, min(flash_kv_heads, num_key_value_heads - 1))
            tropical_kv_heads = num_key_value_heads - flash_kv_heads

            self.flash_kv_heads = flash_kv_heads
            self.tropical_kv_heads = tropical_kv_heads
            self.flash_heads = flash_kv_heads * self.q_per_kv
            self.tropical_heads = tropical_kv_heads * self.q_per_kv
            self.shared_kv = False
        else:
            # MQA case: both branches share the one KV head.
            flash_heads = int(round(num_heads * flash_fraction))
            flash_heads = max(1, min(flash_heads, num_heads - 1))

            self.flash_heads = flash_heads
            self.tropical_heads = num_heads - flash_heads
            self.flash_kv_heads = 1
            self.tropical_kv_heads = 1
            self.shared_kv = True

    def _branch_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [bs, seq, heads, head_dim]
        x_f = x.to(torch.float32)
        var = x_f.square().mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.branch_norm_eps)
        return x_f.to(x.dtype)
    
    def _flash_or_sdpa(
        self,
        query: torch.Tensor,   # [bs, seq, heads, dim]
        key: torch.Tensor,     # [bs, seq, heads, dim]
        value: torch.Tensor,   # [bs, seq, heads, dim]
    ) -> torch.Tensor:
        global _FLASH_ATTN_ENABLED

        attn_output = None
        if _FLASH_ATTN_ENABLED and query.is_cuda and flash_attn_func is not None:
            try:
                attn_output = flash_attn_func(
                    q=query,
                    k=key,
                    v=value,
                    causal=self.causal,
                )
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]
            except RuntimeError as exc:
                message = str(exc).lower()
                if "flashattention only supports ampere gpus or newer" in message:
                    _FLASH_ATTN_ENABLED = False
                else:
                    raise

        if attn_output is None:
            q_t = query.transpose(1, 2)   # [bs, heads, seq, dim]
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)
            attn_output = scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=None,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=self.causal,
            ).transpose(1, 2)

        return attn_output

    def _tropical_attention(
        self,
        query: torch.Tensor,   # [bs, q_len, heads, dim]
        key: torch.Tensor,     # [bs, k_len, heads, dim]
        value: torch.Tensor,   # [bs, k_len, heads, dim]
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.valuation_map:
            q_t = torch.log1p(F.relu(query.to(torch.float32)))
            k_t = torch.log1p(F.relu(key.to(torch.float32)))
            v_t = torch.log1p(F.relu(value.to(torch.float32)))
            diff = q_t.unsqueeze(2) - k_t.unsqueeze(1)  # [bs, q, k, h, d]
        else:
            q_t = query.to(torch.float32)
            k_t = key.to(torch.float32)
            v_t = value.to(torch.float32)
            diff = q_t.unsqueeze(2) - k_t.unsqueeze(1)

        max_diff = diff.amax(dim=-1)                    # [bs, q, k, h]
        min_diff = diff.amin(dim=-1)                    # [bs, q, k, h]
        attn_scores = -(max_diff - min_diff)            # [bs, q, k, h]

        if self.causal:
            q_len = attn_scores.size(1)
            k_len = attn_scores.size(2)
            causal_mask = torch.ones(
                q_len, k_len, device=attn_scores.device, dtype=torch.bool
            ).triu(1)
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(-1),
                torch.finfo(attn_scores.dtype).min,
            )

        sum_sv = attn_scores.unsqueeze(-1) + v_t.unsqueeze(1)   # [bs, q, k, h, d]
        context = sum_sv.max(dim=2).values                      # [bs, q, h, d]

        if self.valuation_map:
            return torch.expm1(context).to(out_dtype)
        return context.to(out_dtype)

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(
            bs,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )

        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.shared_kv:
            flash_q = query[:, :, :self.flash_heads]
            tropical_q = query[:, :, self.flash_heads:]

            flash_k = key.expand(-1, -1, self.flash_heads, -1)
            flash_v = value.expand(-1, -1, self.flash_heads, -1)

            tropical_k = key.expand(-1, -1, self.tropical_heads, -1)
            tropical_v = value.expand(-1, -1, self.tropical_heads, -1)
        else:
            flash_q = query[:, :, :self.flash_heads]
            tropical_q = query[:, :, self.flash_heads:]

            flash_k = key[:, :, :self.flash_kv_heads]
            flash_v = value[:, :, :self.flash_kv_heads]
            tropical_k = key[:, :, self.flash_kv_heads:]
            tropical_v = value[:, :, self.flash_kv_heads:]

            flash_k = repeat_kv(flash_k, self.q_per_kv)
            flash_v = repeat_kv(flash_v, self.q_per_kv)
            tropical_k = repeat_kv(tropical_k, self.q_per_kv)
            tropical_v = repeat_kv(tropical_v, self.q_per_kv)

        flash_out = self._flash_or_sdpa(flash_q, flash_k, flash_v)
        tropical_out = self._tropical_attention(
            tropical_q,
            tropical_k,
            tropical_v,
            out_dtype=hidden_states.dtype,
        )
        flash_out = self._branch_rms_norm(flash_out)
        tropical_out = self._branch_rms_norm(tropical_out)

        flash_out = self.flash_gain * flash_out
        tropical_out = self.tropical_gain * tropical_out

        attn_output = torch.cat([flash_out, tropical_out], dim=2)  # [bs, seq, heads, dim]
        attn_output = attn_output.reshape(bs, seq_len, self.output_size)
        return self.o_proj(attn_output)


@dataclass
class MARMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class MARMConfig(BaseModel):
    batch_size: int
    seq_len: int
    num_puzzle_identifiers: int
    vocab_size: int

    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    num_key_value_heads: Optional[int] = None

    puzzle_emb_ndim: int = 0
    pos_encodings: str = "rope"

    attn_dropout: float = 0.0
    flash_fraction: float = 0.5
    valuation_map: bool = True

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    loops: int = 1
    L_cycles: int = 1
    H_cycles: int = 1
    forward_dtype: str = "bfloat16"
    causal: bool = False


class MARMBlock(nn.Module):
    def __init__(self, config: MARMConfig) -> None:
        super().__init__()
        num_kv_heads = config.num_key_value_heads or config.num_heads

        self.self_attn = MixedAttention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=num_kv_heads,
            causal=config.causal,
            attn_dropout=config.attn_dropout,
            valuation_map=config.valuation_map,
            flash_fraction=config.flash_fraction,
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


class MARM_Inner(nn.Module):
    def __init__(self, config: MARMConfig) -> None:
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
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
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

        self.layers = nn.ModuleList([MARMBlock(self.config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

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

    def empty_carry(self, batch_size: int, device: torch.device | None = None) -> MARMCarry:
        if device is None:
            device = self.init_hidden.device
        return MARMCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: MARMCarry) -> MARMCarry:
        device = carry.current_hidden.device
        reset_flag = reset_flag.to(device)
        init_hidden = self.init_hidden.to(device).view(1, 1, -1)
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            init_hidden,
            carry.current_hidden,
        )
        return replace(carry, current_hidden=new_hidden)

    def forward(
        self,
        carry: MARMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[MARMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
                        hidden_states = hidden_states + input_embeddings
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        probe = getattr(self, "activation_probe", None)
        if probe is not None:
            probe.record_tensor("hidden_states_final", hidden_states)
            probe.record_tensor("new_carry_current_hidden", new_carry.current_hidden)
            probe.record_tensor("output_logits", output)
            probe.record_tensor("q_logits_raw", q_logits)
            probe.record_tensor("q_halt_logits", q_halt_logits)
            probe.record_tensor("q_continue_logits", q_continue_logits)

        return new_carry, output, (q_halt_logits, q_continue_logits)


class MARM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = MARMConfig(**config_dict)
        self.inner = MARM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> MARMCarry:
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device
        base = self.inner.empty_carry(batch_size, device=device)
        return MARMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: MARMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[MARMCarry, Dict[str, torch.Tensor]]:

        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v
            )
            for k, v in carry.current_data.items()
        }

        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                halt_exploration_prob = 0.1
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

        return (
            MARMCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )