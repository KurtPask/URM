import torch

from models.layers import TropicalAttention, TropicalAttentionV3


def main() -> None:
    hidden_size = 64
    num_heads = 8
    head_dim = 8
    seq_len = 16
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        num_key_value_heads=num_heads,
        causal=False,
        q_dropout=0.0,
        k_dropout=0.0,
        v_dropout=0.0,
        tropical_proj=True,
        tropical_qkv_proj=False,
        tropical_norm="none",
        symmetric=True,
    )

    attn_v1 = TropicalAttention(**common_kwargs).to(device)
    attn_v3 = TropicalAttentionV3(**common_kwargs).to(device)
    attn_v3.load_state_dict(attn_v1.state_dict(), strict=True)

    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32, requires_grad=True)

    out_v1, scores_v1 = attn_v1.forward_with_scores(x, cos_sin=None)
    out_v3, scores_v3 = attn_v3.forward_with_scores(x, cos_sin=None)

    out_diff = (out_v1 - out_v3).abs().max().item()
    score_diff = (scores_v1 - scores_v3).abs().max().item()

    print(f"device={device}")
    print(f"max_abs_output_diff={out_diff:.6e}")
    print(f"max_abs_score_diff={score_diff:.6e}")

    loss = out_v3.sum() + scores_v3.sum()
    loss.backward()

    grad_ok = any(p.grad is not None for p in attn_v3.parameters() if p.requires_grad)
    print(f"v3_gradients_present={grad_ok}")


if __name__ == "__main__":
    main()
