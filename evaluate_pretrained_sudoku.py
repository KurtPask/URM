#!/usr/bin/env python3
"""Evaluate a pretrained checkpoint on a Sudoku dataset split.

This wraps ``evaluate_trained_model.evaluate_checkpoint`` and configures
Sudoku-specific evaluator settings so it works with the Sudoku dataset layout.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model on Sudoku")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sudoku-extreme-1k-aug-1000",
        help="Path to Sudoku dataset root (contains train/ and test/)",
    )
    parser.add_argument("--output-dir", type=str, default="eval_results/sudoku", help="Directory to save metrics")
    parser.add_argument("--batch-size", type=int, default=512, help="Global evaluation batch size")
    parser.add_argument("--wandb-project", type=str, default=None, help="Optional Weights & Biases project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional Weights & Biases run name")
    parser.add_argument("--save-predictions", action="store_true", help="Save tensors used during evaluation")

    args = parser.parse_args()

    from evaluate_trained_model import evaluate_checkpoint

    config_overrides = {
        "global_batch_size": args.batch_size,
        "evaluators": [
            {
                "name": "sudoku@Sudoku",
            }
        ],
    }

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config_overrides=config_overrides,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        save_predictions=args.save_predictions,
        loop_offsets=None,
    )


if __name__ == "__main__":
    main()
