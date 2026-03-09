#!/usr/bin/env python3
"""Compare q-head logit distributions between two trained checkpoints.

This script is designed for URM/TARM style checkpoints and extracts
`q_halt_logits` and `q_continue_logits` across evaluation batches.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict

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


def _collect_logits(
    checkpoint_path: Path,
    arch_name: str,
    data_path: str,
    batch_size: int,
    num_batches: int,
    split: str,
) -> Dict[str, np.ndarray]:
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

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{checkpoint_path.name}] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[{checkpoint_path.name}] Unexpected keys when loading checkpoint: {len(unexpected)}")

    halt_values = []
    continue_values = []

    with torch.inference_mode():
        for batch_idx, (_, batch, _) in enumerate(eval_loader):
            if batch_idx >= num_batches:
                break

            batch = {k: v.cuda() for k, v in batch.items()}
            carry = model.initial_carry(batch)

            while True:
                carry, _, _, outputs, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys={"q_halt_logits", "q_continue_logits"},
                )
                halt_values.append(outputs["q_halt_logits"].detach().cpu().reshape(-1).numpy())
                continue_values.append(outputs["q_continue_logits"].detach().cpu().reshape(-1).numpy())
                if all_finish:
                    break

    return {
        "q_halt_logits": np.concatenate(halt_values),
        "q_continue_logits": np.concatenate(continue_values),
    }


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


def _print_summary(model_name: str, logits: Dict[str, np.ndarray]):
    print(f"\n=== {model_name} ===")
    for key, values in logits.items():
        summary = _summarize(values)
        print(f"{key}:")
        for metric_name, metric_value in summary.items():
            print(f"  {metric_name:>6}: {metric_value: .6f}")


def _plot_distributions(
    urm_logits: Dict[str, np.ndarray],
    tarm_logits: Dict[str, np.ndarray],
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
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script because model creation uses CUDA only.")

    print("Collecting URM q-head logits...")
    urm_logits = _collect_logits(
        checkpoint_path=args.urm_checkpoint,
        arch_name=args.urm_arch_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        split=args.split,
    )

    print("Collecting TARM q-head logits...")
    tarm_logits = _collect_logits(
        checkpoint_path=args.tarm_checkpoint,
        arch_name=args.tarm_arch_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        split=args.split,
    )

    _print_summary("URM", urm_logits)
    _print_summary("TARM", tarm_logits)

    _plot_distributions(urm_logits, tarm_logits, args.output_plot, bins=args.bins)
    print(f"\nSaved comparison plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
