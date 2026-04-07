#!/usr/bin/env python3
"""Unified Sudoku checkpoint evaluation runner.

Supports:
1) Evaluating a single checkpoint file.
2) Evaluating the latest checkpoint from every run folder under a checkpoints root.

Results are appended to a timestamped CSV in real time as each checkpoint completes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

STEP_PATTERN = re.compile(r"step_(\d+)\.pt$")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_step_from_name(path: Path) -> Optional[int]:
    match = STEP_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def discover_latest_checkpoints(checkpoints_root: Path) -> List[Path]:
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root does not exist: {checkpoints_root}")

    discovered: List[Path] = []
    for run_dir in sorted(p for p in checkpoints_root.iterdir() if p.is_dir()):
        candidates = [p for p in run_dir.glob("step_*.pt") if parse_step_from_name(p) is not None]
        if not candidates:
            continue
        discovered.append(max(candidates, key=lambda p: parse_step_from_name(p) or -1))

    return discovered


def load_run_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    cfg_path = checkpoint_path.parent / "config.yaml"
    metadata: Dict[str, Any] = {
        "config_path": str(cfg_path),
        "arch": "unknown",
        "train_run_name": "unknown",
    }

    if not cfg_path.exists():
        return metadata

    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        if isinstance(cfg, dict):
            arch = cfg.get("arch")
            if isinstance(arch, dict):
                metadata["arch"] = str(arch.get("name", arch.get("_target_", "dict_arch")))
            elif arch is not None:
                metadata["arch"] = str(arch)
            metadata["train_run_name"] = str(cfg.get("run_name", "unknown"))
    except Exception as exc:  # noqa: BLE001
        metadata["config_read_error"] = str(exc)

    return metadata


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def evaluate_one_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    output_dir: Path,
    batch_size: int,
    max_test_examples_per_set: Optional[int],
    test_example_stride: int,
) -> Dict[str, Any]:
    run_name = checkpoint_path.parent.name
    step = parse_step_from_name(checkpoint_path)
    started_at = utc_now_iso()
    run_metadata = load_run_metadata(checkpoint_path)

    run_output_dir = output_dir / "per_checkpoint" / f"{run_name}__{checkpoint_path.stem}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    base_row: Dict[str, Any] = {
        "run_name": run_name,
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": step,
        "data_path": str(data_path),
        "status": "failed",
        "cell_accuracy": None,
        "exact_accuracy": None,
        "valid_solution_rate": None,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "duration_seconds": None,
        "error": "",
        **run_metadata,
    }

    def _run_eval(batch: int) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            "evaluate_trained_model.py",
            "--checkpoint-path",
            str(checkpoint_path),
            "--data-path",
            str(data_path),
            "--output-dir",
            str(run_output_dir),
            "--batch-size",
            str(batch),
            "--evaluator",
            "sudoku@Sudoku",
            "--eval-log-every-n-batches",
            "0",
        ]
        if max_test_examples_per_set is not None:
            cmd.extend(["--max-test-examples-per-set", str(max_test_examples_per_set)])
        if test_example_stride > 1:
            cmd.extend(["--test-example-stride", str(test_example_stride)])

        env = os.environ.copy()
        env.setdefault("DISABLE_COMPILE", "1")
        return subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)

    try:
        attempts = [batch_size]
        if batch_size > 128:
            attempts.extend([max(128, batch_size // 2), max(64, batch_size // 4)])

        last_result: Optional[subprocess.CompletedProcess[str]] = None
        used_batch = batch_size
        for candidate_batch in attempts:
            result = _run_eval(candidate_batch)
            last_result = result
            used_batch = candidate_batch
            if result.returncode == 0:
                break

            combined_output = (result.stdout + "\n" + result.stderr).lower()
            is_resource_error = any(
                token in combined_output
                for token in ("out of memory", "cuda error", "cublas", "cudnn", "nccl")
            )
            if not is_resource_error:
                break

        if last_result is None or last_result.returncode != 0:
            error_excerpt = ""
            if last_result is not None:
                merged = (last_result.stdout + "\n" + last_result.stderr).strip()
                error_excerpt = merged[-3000:]
            raise RuntimeError(
                f"evaluation subprocess failed (batch_size={used_batch}):\n{error_excerpt}"
            )

        metrics_file = run_output_dir / "metrics.json"
        if not metrics_file.exists():
            raise RuntimeError(f"Missing metrics file after evaluation: {metrics_file}")

        import json

        metrics = json.loads(metrics_file.read_text())
        base_row["cell_accuracy"] = metrics.get("Sudoku/cell_accuracy")
        base_row["exact_accuracy"] = metrics.get("Sudoku/exact_accuracy")
        base_row["valid_solution_rate"] = metrics.get("Sudoku/valid_solution_rate")
        base_row["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        base_row["error"] = str(exc)

    completed = dt.datetime.now(dt.timezone.utc)
    started = dt.datetime.fromisoformat(started_at)
    base_row["completed_at_utc"] = completed.isoformat()
    base_row["duration_seconds"] = round((completed - started).total_seconds(), 3)
    return base_row


def append_csv_row(csv_path: Path, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def build_csv_path(output_dir: Path, explicit_csv_path: Optional[Path]) -> Path:
    if explicit_csv_path is not None:
        return explicit_csv_path
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"sudoku_eval_{ts}.csv"


def print_summary_table(rows: List[Dict[str, Any]]) -> None:
    print("\nSummary (Sudoku test set):")
    print("run_name | checkpoint_step | status | cell_accuracy | exact_accuracy")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['run_name']} | {row['checkpoint_step']} | {row['status']} | "
            f"{format_metric(row['cell_accuracy'])} | {format_metric(row['exact_accuracy'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sudoku checkpoints and stream results into CSV.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Single checkpoint to evaluate (.pt)")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints"),
        help="Root containing run directories (each with step_*.pt files)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/sudoku-extreme-1k-aug-1000"),
        help="Sudoku dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/sudoku"),
        help="Directory for per-run outputs and CSV reports",
    )
    parser.add_argument("--csv-path", type=Path, default=None, help="Optional explicit CSV path")
    parser.add_argument("--batch-size", type=int, default=512, help="Global evaluation batch size")
    parser.add_argument(
        "--max-test-examples-per-set",
        type=int,
        default=None,
        help="Optional cap on examples per test set for faster checkpoint sweeps.",
    )
    parser.add_argument(
        "--test-example-stride",
        type=int,
        default=1,
        help="Use every Nth test example (N>1 for fast subsampled evaluation).",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.data_path}")
    if not (args.data_path / "test" / "dataset.json").exists():
        raise FileNotFoundError(f"Missing required file: {args.data_path / 'test' / 'dataset.json'}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.test_example_stride <= 0:
        raise ValueError("--test-example-stride must be >= 1.")

    if args.checkpoint_path is not None:
        checkpoints = [args.checkpoint_path]
    else:
        checkpoints = discover_latest_checkpoints(args.checkpoints_root)

    if not checkpoints:
        raise RuntimeError("No checkpoints found to evaluate.")

    csv_path = build_csv_path(args.output_dir, args.csv_path)
    rows: List[Dict[str, Any]] = []
    fieldnames = [
        "run_name",
        "train_run_name",
        "arch",
        "checkpoint_file",
        "checkpoint_path",
        "checkpoint_step",
        "config_path",
        "data_path",
        "status",
        "cell_accuracy",
        "exact_accuracy",
        "valid_solution_rate",
        "started_at_utc",
        "completed_at_utc",
        "duration_seconds",
        "error",
    ]

    print(f"Evaluating {len(checkpoints)} checkpoint(s) against Sudoku test split.")
    print(f"Live CSV: {csv_path}")

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        if not checkpoint_path.exists():
            row = {
                "run_name": checkpoint_path.parent.name,
                "train_run_name": "unknown",
                "arch": "unknown",
                "checkpoint_file": checkpoint_path.name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": parse_step_from_name(checkpoint_path),
                "config_path": str(checkpoint_path.parent / "config.yaml"),
                "data_path": str(args.data_path),
                "status": "failed",
                "cell_accuracy": None,
                "exact_accuracy": None,
                "valid_solution_rate": None,
                "started_at_utc": utc_now_iso(),
                "completed_at_utc": utc_now_iso(),
                "duration_seconds": 0.0,
                "error": "checkpoint file not found",
            }
            rows.append(row)
            append_csv_row(csv_path, fieldnames, row)
            print(f"\n[{index}/{len(checkpoints)}] Skipping missing checkpoint: {checkpoint_path}")
            continue

        print(f"\n[{index}/{len(checkpoints)}] Evaluating {checkpoint_path}")
        row = evaluate_one_checkpoint(
            checkpoint_path=checkpoint_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_test_examples_per_set=args.max_test_examples_per_set,
            test_example_stride=args.test_example_stride,
        )
        rows.append(row)
        append_csv_row(csv_path, fieldnames, row)
        print(
            f"Completed {checkpoint_path.name} | status={row['status']} | "
            f"cell_acc={format_metric(row['cell_accuracy'])} | exact_acc={format_metric(row['exact_accuracy'])}"
        )
        if row["error"]:
            print(f"  Error: {row['error']}")

    print_summary_table(rows)
    print(f"\nCSV written to: {csv_path}")


if __name__ == "__main__":
    main()
