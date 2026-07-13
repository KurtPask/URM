from typing import Tuple, Optional
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
import einops
import math

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

from models.common import trunc_normal_init_

try:
    from tropical_gemm.pytorch import (
        tropical_maxplus_matmul,
        tropical_maxplus_matmul_gpu,
        tropical_maxplus_matmul_batched,
        tropical_minplus_matmul_batched,
        GPU_AVAILABLE,
        _DLPACK_AVAILABLE,
        _BATCHED_DLPACK_AVAILABLE,
    )
    _TROPICAL_GEMM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    tropical_maxplus_matmul = None
    tropical_maxplus_matmul_gpu = None
    tropical_maxplus_matmul_batched = None
    tropical_minplus_matmul_batched = None
    GPU_AVAILABLE = False
    _DLPACK_AVAILABLE = False
    _BATCHED_DLPACK_AVAILABLE = False
    _TROPICAL_GEMM_IMPORT_ERROR = exc


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _require_tropical_gemm() -> None:
    if tropical_maxplus_matmul is None:
        raise RuntimeError(
            "tropical-gemm is required for this path. Install tropical-gemm[torch]."
        ) from _TROPICAL_GEMM_IMPORT_ERROR


def _tg_maxplus_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _require_tropical_gemm()
    a = a.contiguous().to(torch.float32)
    b = b.contiguous().to(torch.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("_tg_maxplus_mm expects 2D tensors.")
    if a.is_cuda or b.is_cuda:
        if not (GPU_AVAILABLE and _DLPACK_AVAILABLE):
            raise RuntimeError("tropical-gemm CUDA path unavailable; missing GPU/DLPACK support.")
        return tropical_maxplus_matmul_gpu(a, b)
    return tropical_maxplus_matmul(a, b)


def _tg_maxplus_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _require_tropical_gemm()
    a = a.contiguous().to(torch.float32)
    b = b.contiguous().to(torch.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("_tg_maxplus_bmm expects 3D tensors.")
    if (a.is_cuda or b.is_cuda) and not _BATCHED_DLPACK_AVAILABLE:
        raise RuntimeError("tropical-gemm batched CUDA path unavailable; missing batched DLPACK support.")
    return tropical_maxplus_matmul_batched(a, b)


def _tg_minplus_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _require_tropical_gemm()
    a = a.contiguous().to(torch.float32)
    b = b.contiguous().to(torch.float32)
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("_tg_minplus_bmm expects 3D tensors.")
    if (a.is_cuda or b.is_cuda) and not _BATCHED_DLPACK_AVAILABLE:
        raise RuntimeError("tropical-gemm batched CUDA path unavailable; missing batched DLPACK support.")
    return tropical_minplus_matmul_batched(a, b)


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Attention kernel (FlashAttention when available, otherwise PyTorch SDPA fallback).
        global _FLASH_ATTN_ENABLED
        attn_output = None
        if _FLASH_ATTN_ENABLED and hidden_states.is_cuda and flash_attn_func is not None:
            try:
                attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
                if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
                    attn_output = attn_output[0]
            except RuntimeError as exc:
                message = str(exc).lower()
                if "flashattention only supports ampere gpus or newer" in message:
                    _FLASH_ATTN_ENABLED = False
                else:
                    raise

        if attn_output is None:
            # Fallback for older GPUs / environments without FlashAttention.
            query_t = query.transpose(1, 2)
            key_t = key.transpose(1, 2)
            value_t = value.transpose(1, 2)
            if query_t.shape[1] != key_t.shape[1]:
                repeat_factor = query_t.shape[1] // key_t.shape[1]
                key_t = key_t.repeat_interleave(repeat_factor, dim=1)
                value_t = value_t.repeat_interleave(repeat_factor, dim=1)
            attn_output = scaled_dot_product_attention(
                query_t,
                key_t,
                value_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=self.causal,
            ).transpose(1, 2)

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


import torch
import torch.nn as nn
import torch.nn.functional as F

# assumes these already exist in your codebase:
# - CastedLinear
# - apply_rotary_pos_emb
# - flash_attn_func

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
    Routes one subset of heads through flash attention and the rest through tropical attention.

    Important:
    - If using GQA/MQA, this splits by whole KV groups so head alignment stays valid.
    - If num_key_value_heads == 1, both branches share the same K/V head.
    """
    def __init__(
        self,
        hidden_size,
        head_dim,
        num_heads,
        num_key_value_heads,
        causal=False,
        attn_dropout=0.0,
        valuation_map: bool = True,
        flash_fraction: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        if not (0.0 < flash_fraction < 1.0):
            raise ValueError("flash_fraction must be between 0 and 1.")

        if num_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads}) for GQA-style routing."
            )

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.output_size = num_heads * head_dim
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.valuation_map = valuation_map

        self.q_per_kv = num_heads // num_key_value_heads

        # Shared projections
        self.qkv_proj = CastedLinear(
            hidden_size,
            (num_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

        # Split heads in a GQA-safe way.
        # If kv_heads > 1, split by kv groups.
        if num_key_value_heads > 1:
            flash_kv_heads = int(round(num_key_value_heads * flash_fraction))
            flash_kv_heads = max(1, min(flash_kv_heads, num_key_value_heads - 1))

            self.flash_kv_heads = flash_kv_heads
            self.tropical_kv_heads = num_key_value_heads - flash_kv_heads

            self.flash_heads = self.flash_kv_heads * self.q_per_kv
            self.tropical_heads = num_heads - self.flash_heads
            self.shared_kv = False
        else:
            # MQA case: only one KV head, so both branches share it.
            flash_heads = int(round(num_heads * flash_fraction))
            flash_heads = max(1, min(flash_heads, num_heads - 1))

            self.flash_heads = flash_heads
            self.tropical_heads = num_heads - flash_heads

            self.flash_kv_heads = 1
            self.tropical_kv_heads = 1
            self.shared_kv = True

    def _tropical_attention(
        self,
        query: torch.Tensor,   # [bs, q_len, q_heads, head_dim]
        key: torch.Tensor,     # [bs, k_len, q_heads, head_dim]  (already repeated/aligned)
        value: torch.Tensor,   # [bs, k_len, q_heads, head_dim]  (already repeated/aligned)
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.valuation_map:
            q_t = torch.log1p(F.relu(query.to(torch.float32)))
            k_t = torch.log1p(F.relu(key.to(torch.float32)))
            v_t = torch.log1p(F.relu(value.to(torch.float32)))
            diff = q_t.unsqueeze(2) - k_t.unsqueeze(1)   # [bs, q_len, k_len, heads, dim]
        else:
            v_t = value.to(torch.float32)
            diff = query.to(torch.float32).unsqueeze(2) - key.to(torch.float32).unsqueeze(1)

        max_diff = diff.amax(dim=-1)                    # [bs, q_len, k_len, heads]
        min_diff = diff.amin(dim=-1)                    # [bs, q_len, k_len, heads]
        attn_scores = -(max_diff - min_diff)            # [bs, q_len, k_len, heads]

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

        sum_sv = attn_scores.unsqueeze(-1) + v_t.unsqueeze(1)   # [bs, q_len, k_len, heads, dim]
        context = sum_sv.max(dim=2).values                      # [bs, q_len, heads, dim]

        if self.valuation_map:
            return torch.expm1(context).to(out_dtype)
        return context.to(out_dtype)

    def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
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

        # -------------------------
        # Split into flash / tropical branches
        # -------------------------
        if self.shared_kv:
            # MQA: one KV head shared by both branches
            flash_q = query[:, :, :self.flash_heads]
            tropical_q = query[:, :, self.flash_heads:]

            flash_k = key
            flash_v = value

            tropical_k = key.expand(-1, -1, self.tropical_heads, -1)
            tropical_v = value.expand(-1, -1, self.tropical_heads, -1)

        else:
            # GQA-safe split by whole KV groups
            flash_q = query[:, :, :self.flash_heads]
            tropical_q = query[:, :, self.flash_heads:]

            flash_k = key[:, :, :self.flash_kv_heads]
            flash_v = value[:, :, :self.flash_kv_heads]

            tropical_k = key[:, :, self.flash_kv_heads:]
            tropical_v = value[:, :, self.flash_kv_heads:]

            # Tropical branch wants head-aligned K/V
            tropical_k = repeat_kv(tropical_k, self.q_per_kv)
            tropical_v = repeat_kv(tropical_v, self.q_per_kv)

        # -------------------------
        # Flash branch
        # -------------------------
        flash_out = flash_attn_func(
            q=flash_q,
            k=flash_k,
            v=flash_v,
            causal=self.causal,
        )
        if isinstance(flash_out, tuple):  # FA2 / FA3 compatibility
            flash_out = flash_out[0]

        # -------------------------
        # Tropical branch
        # -------------------------
        tropical_out = self._tropical_attention(
            tropical_q,
            tropical_k,
            tropical_v,
            out_dtype=hidden_states.dtype,
        )

        # Concatenate heads back together
        attn_output = torch.cat([flash_out, tropical_out], dim=2)   # [bs, seq, num_heads, head_dim]
        attn_output = attn_output.reshape(bs, seq_len, self.output_size)

        return self.o_proj(attn_output)


class TropicalAttentionV2(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0, valuation_map: bool = True, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.valuation_map = valuation_map

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim) # [bs, seq_len, num_heads + 2 * num_key_value_heads, head_dim]
        query = qkv[:, :, :self.num_heads] # [bs, seq_len, num_heads, head_dim]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads] # [bs, seq_len, num_key_value_heads, head_dim]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:] # [bs, seq_len, num_key_value_heads, head_dim]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin) # each: [bs, seq_len, num_heads, head_dim]

        # flash attn
        # attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        # if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
        #     attn_output = attn_output[0]

        ######
        if self.valuation_map:
            q_t = torch.log1p(F.relu(query.to(torch.float32))) # [bs, seq_len, num_heads, head_dim]
            k_t = torch.log1p(F.relu(key.to(torch.float32)))# [bs, seq_len, num_heads, head_dim]
            v_t = torch.log1p(F.relu(value.to(torch.float32))) # [bs, seq_len, num_heads, head_dim]

            diff = q_t.unsqueeze(2) - k_t.unsqueeze(1) # [bs, q_seq_len, k_seq_len, num_heads, head_dim]
        else:
            diff = query.unsqueeze(2) - key.unsqueeze(1) # [bs, q_seq_len, k_seq_len, num_heads, head_dim]
        max_diff = diff.amax(dim=-1) # [bs, q_seq_len, k_seq_len, num_heads]
        min_diff = diff.amin(dim=-1) # [bs, q_seq_len, k_seq_len, num_heads]
        attn_scores = -(max_diff - min_diff) # [bs, q_seq_len, k_seq_len, num_heads]
        sum_sv = attn_scores.unsqueeze(-1) + v_t.unsqueeze(1) if self.valuation_map else attn_scores.unsqueeze(-1) + value.unsqueeze(1) # [bs, q_seq_len, k_seq_len, num_heads, head_dim]
        context = sum_sv.max(dim=2).values # [bs, q_seq_len, num_heads, head_dim]
        if self.valuation_map:  
            attn_output = torch.expm1(context).to(hidden_states.dtype) # [bs, q_seq_len, num_heads, head_dim]
        else:
            attn_output = context.to(hidden_states.dtype) # [bs, q_seq_len, num_heads, head_dim]
        ######

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # [bs, seq_len, num_heads * head_dim]
        return self.o_proj(attn_output) # [bs, seq_len, hidden_size]
 
class TropicalLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        diag_init: float = 0.0,
        offdiag_init: float = -9.0,
        init_jitter_std: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize near tropical identity: diagonal ~ 0, off-diagonal very negative.
        weight = torch.full((output_dim, input_dim), offdiag_init, dtype=torch.float32)
        diag_size = min(input_dim, output_dim)
        diag_idx = torch.arange(diag_size)
        weight[diag_idx, diag_idx] = diag_init
        if init_jitter_std > 0:
            weight = weight + init_jitter_std * torch.randn_like(weight)

        self.W = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        try:
            x2 = x.reshape(-1, self.input_dim)
            y2 = _tg_maxplus_mm(x2, self.W.T)
            return y2.reshape(*x.shape[:-1], self.output_dim).to(x_dtype)
        except Exception:
            x_expanded = x.unsqueeze(-2)
            W_expanded = self.W.unsqueeze(0)
            Wx = x_expanded + W_expanded
            y, _ = torch.max(Wx, dim=-1)
            return y


class DeepSet(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = dim if hidden_dim is None else hidden_dim
        self.phi = nn.Sequential(
            CastedLinear(dim, hidden, bias=True),
            nn.SiLU(),
            CastedLinear(hidden, dim, bias=True),
        )
        self.rho = nn.Sequential(
            CastedLinear(dim, hidden, bias=True),
            nn.SiLU(),
            CastedLinear(hidden, dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim] -> [batch, 1, dim]
        phi_x = self.phi(x)
        pooled = phi_x.mean(dim=1)
        out = self.rho(pooled)
        return out.unsqueeze(1)


class TropicalAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        causal: bool = False,
        attn_dropout: float = 0.0,
        q_dropout: float = 0.0,
        k_dropout: float = 0.0,
        v_dropout: float = 0.0,
        tropical_proj: bool = True,
        tropical_qkv_proj: bool = False,
        tropical_norm: str = "none",
        symmetric: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if head_dim != hidden_size // num_heads:
            raise ValueError("head_dim must match hidden_size // num_heads.")

        self.d_k = hidden_size // num_heads
        self.n_heads = num_heads
        self.tropical_proj = tropical_proj
        self.tropical_qkv_proj = tropical_qkv_proj
        if isinstance(tropical_norm, bool):
            tropical_norm = "learnable" if tropical_norm else "none"
        if tropical_norm not in ("none", "max", "learnable"):
            raise ValueError("tropical_norm must be one of: 'none', 'max', 'learnable'.")
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.attn_dropout = attn_dropout

        self.out = CastedLinear(hidden_size, hidden_size, bias=False)
        self.q_dropout = nn.Dropout(q_dropout)
        self.k_dropout = nn.Dropout(k_dropout)
        self.v_dropout = nn.Dropout(v_dropout)

        if self.tropical_qkv_proj:
            self.query_proj = TropicalLinear(hidden_size, hidden_size)
            self.key_proj = TropicalLinear(hidden_size, hidden_size)
            self.value_proj = TropicalLinear(hidden_size, hidden_size)

        if self.tropical_proj:
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm == "learnable":
            self.deepset = DeepSet(hidden_size)

    def normalize_tropical(self, x: torch.Tensor) -> torch.Tensor:
        if self.tropical_norm == "none":
            return x
        if self.tropical_norm == "max":
            return x - x.max(dim=-1, keepdim=True).values
        if self.tropical_norm == "learnable":
            return x - self.deepset(x)
        raise ValueError("Invalid tropical_norm.")

    def _apply_tropical_feature_dropout(
        self, diff: torch.Tensor, keep_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dropout on feature coordinates for symmetric tropical distance.

        Standard nn.Dropout on q/k is not appropriate here because zeroed coordinates
        can still be selected by max/min reductions and thus still influence both the
        forward tropical distance and backward gradients. Instead, we mask `diff`
        directly before reduction and exclude dropped coordinates from selection by
        setting them to +/-inf in the max/min branches.
        """
        q_keep = 1.0 - self.q_dropout.p
        k_keep = 1.0 - self.k_dropout.p
        # A feature coordinate is considered dropped if either q or k would have
        # dropped it under independent Bernoulli masks.
        keep_prob = q_keep * k_keep
        drop_prob = 1.0 - keep_prob

        if (not self.training) or drop_prob <= 0.0:
            return diff, diff

        if keep_mask is None:
            keep_mask = torch.rand_like(diff) < keep_prob
        else:
            keep_mask = keep_mask.to(device=diff.device, dtype=torch.bool)
            if keep_mask.shape != diff.shape:
                raise ValueError("keep_mask must have the same shape as diff.")

        # Ensure numerical safety: if all coordinates are dropped for a reduction row,
        # restore that row to the unmasked diff so reductions remain finite.
        all_dropped = ~keep_mask.any(dim=-1, keepdim=True)
        if all_dropped.any():
            keep_mask = torch.where(all_dropped, torch.ones_like(keep_mask), keep_mask)

        diff_max = diff.masked_fill(~keep_mask, float("-inf"))
        diff_min = diff.masked_fill(~keep_mask, float("inf"))
        return diff_max, diff_min

    def forward_with_scores(self, x: torch.Tensor, cos_sin: Optional[CosSin] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        x_pos = torch.log1p(F.relu(x))
        if self.tropical_norm != "none":
            x_pos = self.normalize_tropical(x_pos)
        if self.tropical_qkv_proj:
            q = self.query_proj(x_pos)
            k = self.key_proj(x_pos)
            v = self.value_proj(x_pos)
        else:
            q = x_pos
            k = x_pos
            v = x_pos

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        B = batch_size * self.n_heads
        q = q.reshape(B, seq_len, self.d_k)
        k = k.reshape(B, seq_len, self.d_k)
        v = v.reshape(B, seq_len, self.d_k)

        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        if not self.symmetric:
            q = self.q_dropout(q)
            k = self.k_dropout(k)
        v = self.v_dropout(v)

        if self.symmetric:
            diff = q.unsqueeze(2) - k.unsqueeze(1)
            diff_max, diff_min = self._apply_tropical_feature_dropout(diff)
            max_diff = diff_max.amax(dim=-1)
            min_diff = diff_min.amin(dim=-1)
            d_trop = max_diff - min_diff
            attn_scores = -d_trop
        else:
            diff = q.unsqueeze(2) - k.unsqueeze(1)
            sum_diff = diff.sum(dim=-1)
            min_diff = diff.amin(dim=-1)
            n = q.size(-1)
            attn_scores = -(sum_diff - n * min_diff)

        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)
        context = sum_sv.max(dim=2).values

        context = (
            context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, -1)
        )

        context = torch.expm1(context)
        output = self.out(context)

        return output, attn_scores

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_scores(hidden_states, cos_sin=cos_sin)
        return output


class TropicalAttentionV3(TropicalAttention):
    def forward_with_scores(self, x: torch.Tensor, cos_sin: Optional[CosSin] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.q_dropout.p != 0.0 or self.k_dropout.p != 0.0 or self.v_dropout.p != 0.0:
            raise ValueError("TropicalAttentionV3 currently requires q_dropout=k_dropout=v_dropout=0.0.")
        if not self.symmetric:
            raise ValueError("TropicalAttentionV3 currently supports only symmetric Hilbert tropical attention.")

        batch_size, seq_len, _ = x.size()
        x_pos = torch.log1p(F.relu(x))
        if self.tropical_norm != "none":
            x_pos = self.normalize_tropical(x_pos)

        if self.tropical_qkv_proj:
            q = self.query_proj(x_pos)
            k = self.key_proj(x_pos)
            v = self.value_proj(x_pos)
        else:
            q, k, v = x_pos, x_pos, x_pos

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.permute(0, 2, 1, 3).reshape(batch_size * self.n_heads, seq_len, self.d_k)
        k = k.permute(0, 2, 1, 3).reshape(batch_size * self.n_heads, seq_len, self.d_k)
        v = v.permute(0, 2, 1, 3).reshape(batch_size * self.n_heads, seq_len, self.d_k)

        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        q = q.to(torch.float32).contiguous()
        k = k.to(torch.float32).contiguous()
        v = v.to(torch.float32).contiguous()

        neg_k_t = (-k).transpose(1, 2).contiguous()
        max_diff = _tg_maxplus_bmm(q, neg_k_t)
        min_diff = _tg_minplus_bmm(q, neg_k_t)
        attn_scores = -(max_diff - min_diff)

        context = _tg_maxplus_bmm(attn_scores, v)
        context = (
            context.reshape(batch_size, self.n_heads, seq_len, self.d_k)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, -1)
        )
        context = torch.expm1(context)
        output = self.out(context.to(x.dtype))
        return output, attn_scores

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_scores(hidden_states, cos_sin=cos_sin)
        return output


class ToricBoundaryRouter(nn.Module):
    """Braid-fan quotient and calibrated boundary gate for a whole block.

    The router does not own the reasoner.  A caller should:

      1. call ``project_input(x)`` to obtain ``pi_sigma(x)``, ``alpha`` and the
         fan state selected by ``x``;
      2. run the full block on both ``x`` and ``pi_sigma(x)``;
      3. call ``project_output(boundary, state)`` so the boundary branch cannot
         reintroduce the killed cone directions;
      4. blend with ``blend(interior, boundary, alpha)``.

    This is intentionally block-level.  Wrapping only attention lets residual and
    MLP paths bypass the quotient, which is not a compactification of the block.
    """

    def __init__(
        self,
        hidden_size: int,
        route_dim: int = 4,
        gate_tau: float = 1.0,
        gate_temp: float = 0.5,
        dir_tau: float = 0.25,
        dir_temp: float = 0.1,
        learn_thresholds: bool = False,
        gate_mode: str = "calibrated",
        gate_quantile: float = 0.95,
        gate_ema: float = 0.05,
        gate_stat: str = "route_norm",
        pinv_ridge: float = 1e-3,
        partition_mode: str = "hard",
    ):
        super().__init__()
        if route_dim < 2:
            raise ValueError("route_dim must be >= 2 for a non-trivial braid fan.")
        if route_dim > hidden_size:
            raise ValueError("route_dim must be <= hidden_size for a stable lifted quotient.")
        if gate_mode not in {"calibrated", "frozen", "force_interior", "force_boundary"}:
            raise ValueError("gate_mode must be: calibrated, frozen, force_interior, or force_boundary.")
        if gate_stat not in {"route_norm", "route_mahalanobis"}:
            raise ValueError("gate_stat must be: route_norm or route_mahalanobis.")
        if partition_mode != "hard":
            raise ValueError("Only hard braid-fan partitions are implemented.")

        self.hidden_size = hidden_size
        self.route_dim = route_dim
        self.gate_mode = gate_mode
        self.gate_quantile = float(gate_quantile)
        self.gate_ema = float(gate_ema)
        self.gate_margin = float(gate_tau)
        self.gate_temp_multiplier = float(gate_temp)
        self.gate_stat = gate_stat
        self.pinv_ridge = float(pinv_ridge)
        self.partition_mode = partition_mode

        # Routing map W_route : R^H -> R^r.  The fan lives after centering
        # modulo R*1 in route space.
        self.route = CastedLinear(hidden_size, route_dim, bias=False)

        def _inv_softplus(value: float) -> torch.Tensor:
            value = max(float(value), 1e-4)
            return torch.tensor(math.log(math.expm1(value)))

        gap_tau_t = torch.tensor(float(dir_tau))
        gap_log_temp_t = _inv_softplus(dir_temp)
        if learn_thresholds:
            self.gap_tau = nn.Parameter(gap_tau_t)
            self._gap_log_temp = nn.Parameter(gap_log_temp_t)
        else:
            self.register_buffer("gap_tau", gap_tau_t, persistent=True)
            self.register_buffer("_gap_log_temp", gap_log_temp_t, persistent=True)

        self.register_buffer("gate_threshold", torch.tensor(float(gate_tau)), persistent=True)
        self.register_buffer("gate_scale", torch.tensor(max(float(gate_temp), 1e-4)), persistent=True)
        self.register_buffer("gate_steps", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("gate_feature_mean", torch.zeros(route_dim), persistent=True)
        self.register_buffer("gate_feature_var", torch.ones(route_dim), persistent=True)

        # Diagnostics (populated each forward; detached, for probing only).
        self.last_alpha_mean: Optional[torch.Tensor] = None
        self.last_block_count_mean: Optional[torch.Tensor] = None
        self.last_gap_boundary_mean: Optional[torch.Tensor] = None
        self.last_quotient_fraction_mean: Optional[torch.Tensor] = None
        self.last_output_leakage_mean: Optional[torch.Tensor] = None
        self.last_route_condition: Optional[torch.Tensor] = None

    def _center_route(self, x: torch.Tensor) -> torch.Tensor:
        u = self.route(x.to(torch.float32))
        return u - u.mean(dim=-1, keepdim=True)

    def _fan_state(self, u: torch.Tensor) -> dict:
        """Select the braid-fan face by sorted gaps.

        Large sorted gaps create block boundaries; small gaps remain in the same
        ordered block.  The block IDs are returned in original route-coordinate
        order, so the same quotient can be applied to outputs.
        """
        gap_temp = F.softplus(self._gap_log_temp).clamp_min(1e-4)
        sorted_u, sort_idx = torch.sort(u, dim=-1)
        gaps = sorted_u[..., 1:] - sorted_u[..., :-1]
        boundary_soft = torch.sigmoid((gaps - self.gap_tau) / gap_temp)

        with torch.no_grad():
            boundary_hard = gaps > self.gap_tau
            block_id_sorted = torch.cat(
                [
                    torch.zeros_like(boundary_hard[..., :1], dtype=torch.long),
                    boundary_hard.to(torch.long).cumsum(dim=-1),
                ],
                dim=-1,
            )
            block_id = torch.empty_like(block_id_sorted)
            block_id.scatter_(dim=-1, index=sort_idx, src=block_id_sorted)
            block_count = block_id_sorted[..., -1].to(torch.float32) + 1.0

        return {
            "block_id": block_id,
            "block_count": block_count,
            "boundary_soft": boundary_soft,
            "boundary_hard": boundary_hard,
        }

    def _block_level(self, values: torch.Tensor, block_id: torch.Tensor) -> torch.Tensor:
        """Return the per-block constant component in route space."""
        original_shape = values.shape
        flat_values = values.reshape(-1, self.route_dim)
        flat_ids = block_id.reshape(-1, self.route_dim)
        membership = F.one_hot(flat_ids, num_classes=self.route_dim).to(flat_values.dtype)
        counts = membership.sum(dim=1).clamp_min(1.0)
        sums = torch.einsum("nrc,nr->nc", membership, flat_values)
        means = sums / counts
        level = torch.gather(means, dim=1, index=flat_ids)
        return level.reshape(original_shape)

    def _lift_route(self, route_values: torch.Tensor) -> torch.Tensor:
        """Lift route-space cone components through a ridge pseudo-inverse."""
        weight = self.route.weight.to(torch.float32)  # [r, H]
        gram = weight @ weight.transpose(0, 1)
        eye = torch.eye(self.route_dim, dtype=gram.dtype, device=gram.device)
        ridge_scale = gram.diagonal().mean().detach().clamp_min(1e-6)
        gram = gram + self.pinv_ridge * ridge_scale * eye

        flat = route_values.reshape(-1, self.route_dim)
        coeff = torch.linalg.solve(gram, flat.transpose(0, 1)).transpose(0, 1)
        lifted = coeff @ weight

        with torch.no_grad():
            singular = torch.linalg.svdvals(weight)
            self.last_route_condition = (
                singular.max() / singular.min().clamp_min(1e-6)
            ).detach()

        return lifted.reshape(*route_values.shape[:-1], self.hidden_size)

    def _gate_statistic(self, u: torch.Tensor) -> torch.Tensor:
        if self.gate_stat == "route_mahalanobis":
            mean = self.gate_feature_mean.to(u.device, dtype=u.dtype)
            var = self.gate_feature_var.to(u.device, dtype=u.dtype).clamp_min(1e-6)
            z = (u - mean.view(*([1] * (u.ndim - 1)), -1)) / torch.sqrt(var.view(*([1] * (u.ndim - 1)), -1))
            return torch.linalg.vector_norm(z, dim=-1, keepdim=True) / math.sqrt(self.route_dim)
        return torch.linalg.vector_norm(u, dim=-1, keepdim=True)

    @torch.no_grad()
    def _update_gate_calibration(self, u: torch.Tensor, stat: torch.Tensor) -> None:
        flat_u = u.detach().reshape(-1, self.route_dim)
        flat_stat = stat.detach().reshape(-1)
        if flat_stat.numel() == 0:
            return

        batch_mean = flat_u.mean(dim=0)
        batch_var = flat_u.var(dim=0, unbiased=False).clamp_min(1e-6)
        q = min(max(self.gate_quantile, 0.5), 0.999)
        hi = torch.quantile(flat_stat, q)
        med = torch.quantile(flat_stat, 0.5)
        robust_scale = (hi - med).abs().clamp_min(1e-3)
        threshold = hi + self.gate_margin * robust_scale
        scale = robust_scale

        if int(self.gate_steps.item()) == 0:
            self.gate_feature_mean.copy_(batch_mean)
            self.gate_feature_var.copy_(batch_var)
            self.gate_threshold.copy_(threshold)
            self.gate_scale.copy_(scale)
        else:
            ema = min(max(self.gate_ema, 0.0), 1.0)
            self.gate_feature_mean.lerp_(batch_mean, ema)
            self.gate_feature_var.lerp_(batch_var, ema)
            self.gate_threshold.lerp_(threshold, ema)
            self.gate_scale.lerp_(scale, ema)
        self.gate_steps.add_(1)

    def _alpha(self, u: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "force_interior":
            return torch.zeros(*u.shape[:-1], 1, dtype=torch.float32, device=u.device)
        if self.gate_mode == "force_boundary":
            return torch.ones(*u.shape[:-1], 1, dtype=torch.float32, device=u.device)

        stat = self._gate_statistic(u)
        if self.training and self.gate_mode == "calibrated":
            self._update_gate_calibration(u, stat)

        temp = (self.gate_temp_multiplier * self.gate_scale.to(u.device)).clamp_min(1e-4)
        threshold = self.gate_threshold.to(u.device)
        return torch.sigmoid((stat - threshold) / temp)

    def project_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        xf = x.to(torch.float32)
        u = self._center_route(xf)
        state = self._fan_state(u)
        block_level = self._block_level(u, state["block_id"])
        removed = self._lift_route(block_level)
        x_quot = xf - removed
        alpha = self._alpha(u)

        denom = torch.linalg.vector_norm(u, dim=-1).clamp_min(1e-6)
        quot_frac = torch.linalg.vector_norm(block_level, dim=-1) / denom
        self.last_alpha_mean = alpha.detach().mean()
        self.last_block_count_mean = state["block_count"].detach().mean()
        self.last_gap_boundary_mean = state["boundary_soft"].detach().mean()
        self.last_quotient_fraction_mean = quot_frac.detach().mean()
        self.last_output_leakage_mean = None
        return x_quot.to(x.dtype), alpha, state

    def project_output(self, y: torch.Tensor, state: dict) -> torch.Tensor:
        yf = y.to(torch.float32)
        v = self._center_route(yf)
        block_level = self._block_level(v, state["block_id"])
        projected = yf - self._lift_route(block_level)

        with torch.no_grad():
            after = self._center_route(projected)
            after_level = self._block_level(after, state["block_id"])
            denom = torch.linalg.vector_norm(v, dim=-1).clamp_min(1e-6)
            leakage = torch.linalg.vector_norm(after_level, dim=-1) / denom
            self.last_output_leakage_mean = leakage.detach().mean()

        return projected.to(y.dtype)

    @staticmethod
    def blend(interior: torch.Tensor, boundary: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha = alpha.to(interior.dtype)
        return (1.0 - alpha) * interior + alpha * boundary


class TropicalAttentionV4(TropicalAttentionV2):
    """
    TropicalAttentionV2-style module that uses tropical-gemm batched kernels
    for the tropical attention computation only.
    """

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if self.num_key_value_heads != self.num_heads:
            if self.num_heads % self.num_key_value_heads != 0:
                raise ValueError("num_heads must be divisible by num_key_value_heads for TropicalAttentionV4.")
            rep = self.num_heads // self.num_key_value_heads
            key = repeat_kv(key, rep)
            value = repeat_kv(value, rep)

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if self.valuation_map:
            q_t = torch.log1p(F.relu(query.to(torch.float32)))
            k_t = torch.log1p(F.relu(key.to(torch.float32)))
            v_t = torch.log1p(F.relu(value.to(torch.float32)))
        else:
            q_t = query.to(torch.float32)
            k_t = key.to(torch.float32)
            v_t = value.to(torch.float32)

        q_b = q_t.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim).contiguous()
        k_b = k_t.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim).contiguous()
        v_b = v_t.permute(0, 2, 1, 3).reshape(batch_size * self.num_heads, seq_len, self.head_dim).contiguous()

        neg_k_t = (-k_b).transpose(1, 2).contiguous()
        max_diff = _tg_maxplus_bmm(q_b, neg_k_t)
        min_diff = _tg_minplus_bmm(q_b, neg_k_t)
        attn_scores = -(max_diff - min_diff)

        context = _tg_maxplus_bmm(attn_scores, v_b)
        context = context.reshape(batch_size, self.num_heads, seq_len, self.head_dim).permute(0, 2, 1, 3)

        if self.valuation_map:
            attn_output = torch.expm1(context).to(hidden_states.dtype)
        else:
            attn_output = context.to(hidden_states.dtype)

        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, mlp_dropout: float = 0.0):
        super().__init__()
        
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.mlp_dropout(F.silu(gate) * up))


class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 2,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter,
            bias=True,
        ).to(dtype=torch.bfloat16)

        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = self.act(gate) * up
        x_conv = self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))
        x_conv = x_conv[..., :up.size(1)]
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_out = self.down_proj(x_conv)

        return x_out


class FullyLinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = round(expansion * hidden_size)

        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class LinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(gate + up)


class SiLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.silu(x)
        return self.down_proj(x)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class ReLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x)
        return self.down_proj(x)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
