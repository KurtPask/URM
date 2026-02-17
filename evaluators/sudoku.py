from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist


class Sudoku:
    required_outputs = {"preds"}

    def __init__(self, data_path: str, eval_metadata, **_: object):
        del data_path  # Unused, kept for a uniform evaluator constructor.
        self.blank_identifier_id = eval_metadata.blank_identifier_id
        self._local_stats: Dict[str, float] = {}

    def begin_eval(self):
        self._local_stats = {
            "num_examples": 0.0,
            "num_exact": 0.0,
            "num_cells": 0.0,
            "num_cells_correct": 0.0,
            "num_valid_solutions": 0.0,
        }

    @staticmethod
    def _is_valid_sudoku_solution(tokenized_board: np.ndarray) -> bool:
        """Validate a predicted Sudoku board.

        The dataset stores digits as token IDs in [2, 10] for values [1, 9].
        """
        board = tokenized_board.reshape(9, 9)

        expected = set(range(2, 11))
        for i in range(9):
            if set(board[i, :].tolist()) != expected:
                return False
            if set(board[:, i].tolist()) != expected:
                return False

        for r0 in (0, 3, 6):
            for c0 in (0, 3, 6):
                if set(board[r0 : r0 + 3, c0 : c0 + 3].reshape(-1).tolist()) != expected:
                    return False

        return True

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        labels = batch["labels"].cpu()
        pred_tokens = preds["preds"].cpu()
        puzzle_identifiers = batch["puzzle_identifiers"].cpu()

        valid_examples = puzzle_identifiers != self.blank_identifier_id
        if not torch.any(valid_examples):
            return

        labels = labels[valid_examples]
        pred_tokens = pred_tokens[valid_examples]

        valid_mask = labels >= 0
        cell_correct = (pred_tokens == labels) & valid_mask
        seq_correct = (cell_correct | ~valid_mask).all(dim=-1)

        self._local_stats["num_examples"] += float(labels.shape[0])
        self._local_stats["num_exact"] += float(seq_correct.sum().item())
        self._local_stats["num_cells"] += float(valid_mask.sum().item())
        self._local_stats["num_cells_correct"] += float(cell_correct.sum().item())

        for solved, prediction in zip(seq_correct.tolist(), pred_tokens.numpy()):
            if solved and self._is_valid_sudoku_solution(prediction):
                self._local_stats["num_valid_solutions"] += 1.0

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Optional[Dict[str, float]]:
        del save_path  # No artifact to save for Sudoku metrics.

        global_stats = None
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            global_stats = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(self._local_stats, global_stats, dst=0, group=group)
        elif rank == 0:
            global_stats = [self._local_stats]

        if rank != 0 or global_stats is None:
            return None

        totals = {
            "num_examples": 0.0,
            "num_exact": 0.0,
            "num_cells": 0.0,
            "num_cells_correct": 0.0,
            "num_valid_solutions": 0.0,
        }
        for worker_stats in global_stats:
            for key in totals:
                totals[key] += float(worker_stats[key])

        num_examples = max(totals["num_examples"], 1.0)
        num_cells = max(totals["num_cells"], 1.0)

        return {
            "Sudoku/cell_accuracy": totals["num_cells_correct"] / num_cells,
            "Sudoku/exact_accuracy": totals["num_exact"] / num_examples,
            "Sudoku/valid_solution_rate": totals["num_valid_solutions"] / num_examples,
        }
