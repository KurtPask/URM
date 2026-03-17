from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json

import torch
import wandb


@dataclass
class ActivationProbeConfig:
    enabled: bool = False
    interval_steps: int = 100
    histogram_interval_steps: int = 500
    raw_save_interval_steps: int = 1000
    save_dir: str = "analysis/activation_probe"
    sample_examples: int = 2
    sample_tokens: int = 16
    sample_hidden_dims: int = 64
    sample_output_vocab_dims: int = 128
    flat_sample_size: int = 65536
    max_hist_points: int = 32768
    rank0_only: bool = True
    log_wandb_histograms: bool = True
    log_wandb_scalars: bool = True
    save_raw_locally: bool = True
    include_hidden_states: bool = True
    include_new_carry: bool = True
    include_output: bool = True
    include_q_logits: bool = True
    include_q_halt: bool = True
    include_q_continue: bool = True
    seed: int = 0


class ActivationProbe:

    def _is_name_enabled(self, name: str) -> bool:
        if name == "hidden_states_final":
            return self.cfg.include_hidden_states
        if name == "new_carry_current_hidden":
            return self.cfg.include_new_carry
        if name == "output_logits":
            return self.cfg.include_output
        if name == "q_logits_raw":
            return self.cfg.include_q_logits
        if name == "q_halt_logits":
            return self.cfg.include_q_halt
        if name == "q_continue_logits":
            return self.cfg.include_q_continue
        return True

    def __init__(self, cfg: ActivationProbeConfig, rank: int):
        self.cfg = cfg
        self.rank = rank
        self._capture_enabled = False
        self._summary_enabled = False
        self._hist_enabled = False
        self._raw_enabled = False
        self._ctx: Dict[str, Any] = {}
        self._tensors: Dict[str, torch.Tensor] = {}

    def set_context(self, *, split: str, global_step: int, run_name: str, arch_name: str) -> None:
        self._ctx = {
            "split": split,
            "global_step": int(global_step),
            "run_name": run_name,
            "arch_name": arch_name,
        }
        step = int(global_step)
        summary_due = step % max(1, self.cfg.interval_steps) == 0
        hist_due = step % max(1, self.cfg.histogram_interval_steps) == 0
        raw_due = step % max(1, self.cfg.raw_save_interval_steps) == 0
        self._capture_enabled = self.cfg.enabled and (summary_due or hist_due or raw_due)
        self._summary_enabled = self._capture_enabled and summary_due and self.cfg.log_wandb_scalars
        self._hist_enabled = self._capture_enabled and hist_due and self.cfg.log_wandb_histograms
        self._raw_enabled = self._capture_enabled and raw_due and self.cfg.save_raw_locally
        self._tensors = {}

    def record_tensor(self, name: str, tensor: Optional[torch.Tensor]) -> None:
        if not self._capture_enabled or tensor is None or not self._is_name_enabled(name):
            return
        self._tensors[name] = tensor.detach()

    def finalize_step(self) -> None:
        if not self._capture_enabled or len(self._tensors) == 0:
            return
        if self.cfg.rank0_only and self.rank != 0:
            self._tensors = {}
            return

        summary: Dict[str, Dict[str, Any]] = {}
        wandb_payload: Dict[str, Any] = {}
        hist_payload: Dict[str, Any] = {}
        raw_samples: Dict[str, torch.Tensor] = {}

        for name, tensor in self._tensors.items():
            with torch.no_grad():
                flat_sample = self._sample_flat(tensor, self.cfg.flat_sample_size)
                stats = self._compute_stats(flat_sample, tensor)
                summary[name] = stats

                if self._summary_enabled:
                    for key, value in stats.items():
                        if isinstance(value, (float, int)):
                            wandb_payload[f"probe/{name}/{key}"] = value

                if self._hist_enabled:
                    hist_sample = self._sample_flat(tensor, self.cfg.max_hist_points)
                    hist_payload[f"probe_hist/{name}"] = wandb.Histogram(hist_sample.numpy())

                if self._raw_enabled:
                    raw_samples[f"{name}/structured"] = self._structured_sample(name, tensor)
                    raw_samples[f"{name}/flat"] = flat_sample.clone()

        step = self._ctx["global_step"]
        if len(wandb_payload):
            wandb.log(wandb_payload, step=step)
        if len(hist_payload):
            wandb.log(hist_payload, step=step)

        if self._raw_enabled:
            run_name = self._ctx["run_name"]
            save_dir = Path(self.cfg.save_dir) / run_name / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(summary, save_dir / "summary.pt")
            torch.save(raw_samples, save_dir / "raw_samples.pt")
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump({"context": self._ctx, "probe_config": asdict(self.cfg)}, f, indent=2)

        self._tensors = {}

    def _sample_flat(self, tensor: torch.Tensor, max_points: int) -> torch.Tensor:
        flat = tensor.reshape(-1)
        if flat.numel() <= max_points:
            return flat.to(torch.float32).cpu()
        g = torch.Generator(device=flat.device)
        g.manual_seed(self.cfg.seed + self._ctx["global_step"])
        idx = torch.randperm(flat.numel(), generator=g, device=flat.device)[:max_points]
        return flat[idx].to(torch.float32).cpu()

    def _compute_stats(self, sample_cpu: torch.Tensor, tensor: torch.Tensor) -> Dict[str, Any]:
        sample = sample_cpu
        finite = torch.isfinite(sample)
        finite_vals = sample[finite]
        if finite_vals.numel() == 0:
            finite_vals = torch.zeros(1, dtype=torch.float32)
        q = torch.quantile(
            finite_vals,
            torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float32),
        )
        return {
            "mean": float(finite_vals.mean().item()),
            "std": float(finite_vals.std(unbiased=False).item()),
            "min": float(finite_vals.min().item()),
            "max": float(finite_vals.max().item()),
            "abs_mean": float(finite_vals.abs().mean().item()),
            "abs_max": float(finite_vals.abs().max().item()),
            "rms": float(torch.sqrt((finite_vals * finite_vals).mean()).item()),
            "p01": float(q[0].item()),
            "p05": float(q[1].item()),
            "p25": float(q[2].item()),
            "p50": float(q[3].item()),
            "p75": float(q[4].item()),
            "p95": float(q[5].item()),
            "p99": float(q[6].item()),
            "finite_frac": float(finite.to(torch.float32).mean().item()),
            "nan_count": int(torch.isnan(sample).sum().item()),
            "inf_count": int(torch.isinf(sample).sum().item()),
            "zero_frac": float((finite_vals == 0).to(torch.float32).mean().item()),
            "positive_frac": float((finite_vals > 0).to(torch.float32).mean().item()),
            "negative_frac": float((finite_vals < 0).to(torch.float32).mean().item()),
            "numel": int(tensor.numel()),
            "sample_numel": int(sample.numel()),
            "shape": [int(x) for x in tensor.shape],
            "dtype": str(tensor.dtype),
        }

    def _structured_sample(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.detach()
        if t.ndim == 3:
            n = min(self.cfg.sample_examples, t.shape[0])
            tok = min(self.cfg.sample_tokens, t.shape[1])
            if "output_logits" in name:
                d = min(self.cfg.sample_output_vocab_dims, t.shape[2])
            else:
                d = min(self.cfg.sample_hidden_dims, t.shape[2])
            return t[:n, :tok, :d].to(torch.float32).cpu().contiguous()
        return t.to(torch.float32).cpu().contiguous()

