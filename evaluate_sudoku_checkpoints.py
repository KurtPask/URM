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
import re
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

    try:
        from evaluate_trained_model import evaluate_checkpoint

        evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
            data_path=str(data_path),
            output_dir=str(run_output_dir),
            config_overrides={
                "global_batch_size": batch_size,
                "evaluators": [{"name": "sudoku@Sudoku"}],
            },
            wandb_project=None,
            wandb_run_name=None,
            save_predictions=False,
            loop_offsets=None,
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
    args = parser.parse_args()

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
        print(f"\n[{index}/{len(checkpoints)}] Evaluating {checkpoint_path}")
        row = evaluate_one_checkpoint(
            checkpoint_path=checkpoint_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
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
