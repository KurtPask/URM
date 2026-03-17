#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def load_step(run_dir: Path, step: int):
    d = run_dir / f"step_{step}"
    return torch.load(d / "summary.pt", map_location="cpu"), torch.load(d / "raw_samples.pt", map_location="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urm-dir", type=Path, required=True)
    ap.add_argument("--tarm-dir", type=Path, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--tensor", type=str, default="hidden_states_final")
    args = ap.parse_args()

    urm_summary, urm_raw = load_step(args.urm_dir, args.step)
    tarm_summary, tarm_raw = load_step(args.tarm_dir, args.step)

    keys = ["mean", "std", "p05", "p50", "p95", "abs_max", "finite_frac"]
    print(f"Tensor: {args.tensor} @ step {args.step}")
    print("metric\tURM\tTARM")
    for k in keys:
        uv = urm_summary[args.tensor].get(k)
        tv = tarm_summary[args.tensor].get(k)
        print(f"{k}\t{uv}\t{tv}")

    u = urm_raw[f"{args.tensor}/flat"].float()
    t = tarm_raw[f"{args.tensor}/flat"].float()
    hist_min = min(u.min().item(), t.min().item())
    hist_max = max(u.max().item(), t.max().item())
    bins = 40
    uh = torch.histc(u, bins=bins, min=hist_min, max=hist_max)
    th = torch.histc(t, bins=bins, min=hist_min, max=hist_max)
    print("\nOverlay histogram counts (first 10 bins):")
    print("URM :", uh[:10].tolist())
    print("TARM:", th[:10].tolist())


if __name__ == "__main__":
    main()
