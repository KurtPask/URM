import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from evaluators.arc import ARC


class ARCRankingScoreTest(unittest.TestCase):
    def _evaluator(self, ranking_score: str):
        tempdir = tempfile.TemporaryDirectory()
        root = Path(tempdir.name)
        (root / "identifiers.json").write_text(json.dumps(["task"]), encoding="utf-8")
        puzzle = {
            "task": {
                "test": [
                    {
                        "input": [[0]],
                        "output": [[0]],
                    }
                ]
            }
        }
        (root / "test_puzzles.json").write_text(json.dumps(puzzle), encoding="utf-8")
        evaluator = ARC(
            data_path=str(root),
            eval_metadata=SimpleNamespace(blank_identifier_id=-1),
            ranking_score=ranking_score,
        )
        return tempdir, evaluator

    @staticmethod
    def _batch_and_predictions(include_score: bool):
        batch = {
            "inputs": torch.full((1, 900), 2, dtype=torch.long),
            "puzzle_identifiers": torch.zeros(1, dtype=torch.long),
        }
        predictions = {
            "preds": torch.full((1, 900), 2, dtype=torch.long),
            "q_halt_logits": torch.full((1,), -8.0),
        }
        if include_score:
            predictions["solution_score_logits"] = torch.ones(1)
        return batch, predictions

    def test_solution_score_is_used_instead_of_halt_probability(self):
        tempdir, evaluator = self._evaluator("solution_score_logits")
        self.addCleanup(tempdir.cleanup)
        batch, predictions = self._batch_and_predictions(include_score=True)
        evaluator.update_batch(batch, predictions)

        stored = next(iter(next(iter(evaluator._local_preds.values())).values()))[0]
        self.assertAlmostEqual(stored[1], float(torch.sigmoid(torch.tensor(1.0))), places=7)

    def test_frequency_ranking_requires_no_model_score(self):
        tempdir, evaluator = self._evaluator("frequency")
        self.addCleanup(tempdir.cleanup)
        batch, predictions = self._batch_and_predictions(include_score=False)
        evaluator.update_batch(batch, predictions)
        stored = next(iter(next(iter(evaluator._local_preds.values())).values()))[0]
        self.assertEqual(stored[1], 0.5)


if __name__ == "__main__":
    unittest.main()
