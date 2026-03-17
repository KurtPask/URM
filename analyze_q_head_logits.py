#!/usr/bin/env python3
"""Compare q-head logit distributions between two trained checkpoints.

This script is designed for URM/TARM style checkpoints and extracts
`q_halt_logits` and `q_continue_logits` across evaluation batches.

It can also capture hidden-layer activations from `model.inner.layers`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Any

os.environ.setdefault("DISABLE_COMPILE", "1")

import numpy as np
import torch


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    return checkpoint_obj


def _unwrap_core_model(model):
    """
    create_model(...) may return a loss wrapper (e.g. ACTLossHead)
    around the actual URM/TARM model. Peel wrappers until we reach
    the underlying architecture module.
    """
    core = model
    seen = set()

    while hasattr(core, "model") and id(core) not in seen:
        seen.add(id(core))
        core = core.model

    return core


def _tensor_summary(x: torch.Tensor) -> Dict[str, Any]:
    x = x.detach().to(torch.float32)
    flat = x.flatten()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "p05": float(torch.quantile(flat, 0.05).item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(x.max().item()),
        "l2": float(torch.linalg.vector_norm(x).item()),
    }


def _sample_hidden_tensor(
    x: torch.Tensor,
    token_limit: int,
    hidden_limit: int,
) -> np.ndarray:
    """
    Reduce [B, S, H] to a smaller sampled block to keep file sizes sane.
    Keeps only first batch element by default.
    """
    x = x.detach().to(torch.float32).cpu()

    if x.ndim != 3:
        return x.numpy()

    b = min(1, x.shape[0])
    s = min(token_limit, x.shape[1])
    h = min(hidden_limit, x.shape[2])
    return x[:b, :s, :h].numpy()


def _collect_logits_and_hidden(
    checkpoint_path: Path,
    arch_name: str,
    data_path: str,
    batch_size: int,
    num_batches: int,
    split: str,
    capture_hidden: bool,
    save_hidden_full: bool,
    hidden_token_limit: int,
    hidden_dim_limit: int,
    hidden_output_dir: Path | None,
) -> Dict[str, Any]:
    from evaluate_trained_model import load_config_from_checkpoint
    from pretrain import create_dataloader, create_model

    config = load_config_from_checkpoint(checkpoint_path)
    config.data_path = data_path
    config.global_batch_size = batch_size
    config.arch.name = arch_name

    eval_loader, eval_metadata = create_dataloader(
        config,
        split,
        test_set_mode=(split == "test"),
        epochs_per_iter=1,
        global_batch_size=batch_size,
        rank=0,
        world_size=1,
    )

    try:
        _, train_metadata = create_dataloader(
            config,
            "train",
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=batch_size,
            rank=0,
            world_size=1,
        )
    except FileNotFoundError:
        train_metadata = eval_metadata

    model, _, _ = create_model(config, train_metadata, rank=0, world_size=1)
    model.eval()

    core_model = _unwrap_core_model(model)

    print(f"[{checkpoint_path.name}] top-level model type: {type(model).__name__}")
    print(f"[{checkpoint_path.name}] core model type: {type(core_model).__name__}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[{checkpoint_path.name}] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[{checkpoint_path.name}] Unexpected keys when loading checkpoint: {len(unexpected)}")

    halt_values: List[np.ndarray] = []
    continue_values: List[np.ndarray] = []

    hidden_stats: List[Dict[str, Any]] = []
    hidden_full_manifest: List[Dict[str, Any]] = []

    hook_handles = []
    current_context = {
        "batch_idx": -1,
        "loop_step": -1,
        "model_name": arch_name,
    }

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            if not capture_hidden:
                return

            if not torch.is_tensor(output):
                return

            record = {
                "model_name": current_context["model_name"],
                "batch_idx": int(current_context["batch_idx"]),
                "loop_step": int(current_context["loop_step"]),
                "layer_idx": int(layer_idx),
                **_tensor_summary(output),
            }
            hidden_stats.append(record)

            if save_hidden_full and hidden_output_dir is not None:
                hidden_output_dir.mkdir(parents=True, exist_ok=True)
                sampled = _sample_hidden_tensor(
                    output,
                    token_limit=hidden_token_limit,
                    hidden_limit=hidden_dim_limit,
                )
                out_name = (
                    f"{arch_name.replace('/', '_').replace('@', '_')}"
                    f"_batch{current_context['batch_idx']:04d}"
                    f"_step{current_context['loop_step']:03d}"
                    f"_layer{layer_idx:03d}.npy"
                )
                out_path = hidden_output_dir / out_name
                np.save(out_path, sampled)
                hidden_full_manifest.append(
                    {
                        "model_name": current_context["model_name"],
                        "batch_idx": int(current_context["batch_idx"]),
                        "loop_step": int(current_context["loop_step"]),
                        "layer_idx": int(layer_idx),
                        "path": str(out_path),
                        "saved_shape": list(sampled.shape),
                    }
                )

        return hook

    if capture_hidden:
        if not hasattr(core_model, "inner"):
            raise AttributeError(
                f"Underlying model of type {type(core_model).__name__} has no .inner module"
            )
        if not hasattr(core_model.inner, "layers"):
            raise AttributeError(
                f"Underlying model.inner of type {type(core_model.inner).__name__} has no .layers"
            )

        for layer_idx, layer in enumerate(core_model.inner.layers):
            hook_handles.append(layer.register_forward_hook(make_hook(layer_idx)))

    with torch.inference_mode():
        for batch_idx, (_, batch, _) in enumerate(eval_loader):
            if batch_idx >= num_batches:
                break

            print(arch_name, batch_idx)
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = model.initial_carry(batch)

            loop_step = 0
            while True:
                current_context["batch_idx"] = batch_idx
                current_context["loop_step"] = loop_step

                model_out = model(
                    carry=carry,
                    batch=batch,
                    return_keys={"q_halt_logits", "q_continue_logits"},
                )

                # Support:
                # 1) carry, _, _, outputs, all_finish
                # 2) (carry, outputs)
                if isinstance(model_out, tuple) and len(model_out) == 5:
                    carry, _, _, outputs, all_finish = model_out
                elif isinstance(model_out, tuple) and len(model_out) == 2:
                    carry, outputs = model_out
                    if not hasattr(carry, "halted") or carry.halted is None:
                        raise RuntimeError(
                            "Model returned 2-tuple but carry.halted is unavailable; cannot determine loop termination."
                        )
                    all_finish = bool(carry.halted.all().item())
                else:
                    raise RuntimeError(
                        f"Unexpected model output structure: type={type(model_out)}, "
                        f"len={len(model_out) if isinstance(model_out, tuple) else 'n/a'}"
                    )

                if "q_halt_logits" not in outputs or "q_continue_logits" not in outputs:
                    raise KeyError(
                        "Expected outputs to include 'q_halt_logits' and 'q_continue_logits'. "
                        f"Found keys: {list(outputs.keys())}"
                    )

                halt_values.append(outputs["q_halt_logits"].detach().cpu().reshape(-1).numpy())
                continue_values.append(outputs["q_continue_logits"].detach().cpu().reshape(-1).numpy())

                loop_step += 1
                if all_finish:
                    break

    for handle in hook_handles:
        handle.remove()

    result: Dict[str, Any] = {
        "q_halt_logits": np.concatenate(halt_values),
        "q_continue_logits": np.concatenate(continue_values),
    }

    if capture_hidden:
        result["hidden_stats"] = hidden_stats
        result["hidden_full_manifest"] = hidden_full_manifest

    return result


def _summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "count": float(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _print_summary(model_name: str, logits: Dict[str, Any]):
    print(f"\n=== {model_name} ===")
    for key in ("q_halt_logits", "q_continue_logits"):
        values = logits[key]
        summary = _summarize(values)
        print(f"{key}:")
        for metric_name, metric_value in summary.items():
            print(f"  {metric_name:>6}: {metric_value: .6f}")

    if "hidden_stats" in logits:
        print(f"hidden_stats records: {len(logits['hidden_stats'])}")


def _plot_distributions(
    urm_logits: Dict[str, Any],
    tarm_logits: Dict[str, Any],
    output_path: Path,
    bins: int,
):
    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib is not installed; skipping histogram plot generation.")
        return

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in (
        (axes[0], "q_halt_logits", "q_halt_logits"),
        (axes[1], "q_continue_logits", "q_continue_logits"),
    ):
        ax.hist(urm_logits[key], bins=bins, alpha=0.45, density=True, label="URM")
        ax.hist(tarm_logits[key], bins=bins, alpha=0.45, density=True, label="TARM")
        ax.set_title(title)
        ax.set_xlabel("Logit value")
        ax.set_ylabel("Density")
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare q-head logits from URM and TARM checkpoints")
    parser.add_argument("--urm-checkpoint", required=True, type=Path)
    parser.add_argument("--tarm-checkpoint", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=str)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-batches", default=20, type=int)
    parser.add_argument("--urm-arch-name", default="urm.urm@URM", type=str)
    parser.add_argument("--tarm-arch-name", default="tarm.tarm@TARM", type=str)
    parser.add_argument("--bins", default=80, type=int)
    parser.add_argument("--output-plot", default="analysis/q_head_logits_comparison.png", type=Path)

    parser.add_argument(
        "--capture-hidden",
        action="store_true",
        help="Capture summaries of outputs from core_model.inner.layers",
    )
    parser.add_argument(
        "--save-hidden-full",
        action="store_true",
        help="Also save sampled hidden tensors as .npy files",
    )
    parser.add_argument(
        "--hidden-token-limit",
        default=16,
        type=int,
        help="Max tokens to save per hidden tensor sample",
    )
    parser.add_argument(
        "--hidden-dim-limit",
        default=64,
        type=int,
        help="Max hidden dims to save per hidden tensor sample",
    )
    parser.add_argument(
        "--hidden-output-dir",
        default="analysis/hidden_layers",
        type=Path,
    )
    parser.add_argument(
        "--hidden-stats-json",
        default="analysis/hidden_layer_stats.json",
        type=Path,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script because model creation uses CUDA only.")

    print("Collecting URM q-head logits...")
    urm_logits = _collect_logits_and_hidden(
        checkpoint_path=args.urm_checkpoint,
        arch_name=args.urm_arch_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        split=args.split,
        capture_hidden=args.capture_hidden,
        save_hidden_full=args.save_hidden_full,
        hidden_token_limit=args.hidden_token_limit,
        hidden_dim_limit=args.hidden_dim_limit,
        hidden_output_dir=(args.hidden_output_dir / "urm") if args.capture_hidden else None,
    )
    _print_summary("URM", urm_logits)

    print("Collecting TARM q-head logits...")
    tarm_logits = _collect_logits_and_hidden(
        checkpoint_path=args.tarm_checkpoint,
        arch_name=args.tarm_arch_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        split=args.split,
        capture_hidden=args.capture_hidden,
        save_hidden_full=args.save_hidden_full,
        hidden_token_limit=args.hidden_token_limit,
        hidden_dim_limit=args.hidden_dim_limit,
        hidden_output_dir=(args.hidden_output_dir / "tarm") if args.capture_hidden else None,
    )
    _print_summary("TARM", tarm_logits)

    _plot_distributions(urm_logits, tarm_logits, args.output_plot, bins=args.bins)
    print(f"\nSaved comparison plot to: {args.output_plot}")

    if args.capture_hidden:
        combined_hidden_stats = {
            "urm": urm_logits.get("hidden_stats", []),
            "tarm": tarm_logits.get("hidden_stats", []),
            "urm_manifest": urm_logits.get("hidden_full_manifest", []),
            "tarm_manifest": tarm_logits.get("hidden_full_manifest", []),
        }
        args.hidden_stats_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.hidden_stats_json, "w") as f:
            json.dump(combined_hidden_stats, f, indent=2)
        print(f"Saved hidden layer stats to: {args.hidden_stats_json}")


if __name__ == "__main__":
    main()