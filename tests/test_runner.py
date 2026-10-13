"""Small CPU checks for independent metric and evaluation bookkeeping."""

import unittest

import numpy as np
import torch
from torch import nn

from experiments.data import WindowDataset
from experiments.run import batch_to_device, evaluate, loader, metrics


class ItemProbability(nn.Module):
    def forward(self, concept_emb, question_emb, responses):
        return question_emb[..., :1]


class RunnerTests(unittest.TestCase):
    def test_auc_ties_match_pairwise_definition(self):
        rng = np.random.default_rng(203)
        for size in (9, 47, 200):
            labels = rng.integers(0, 2, size=size)
            probabilities = rng.integers(0, 8, size=size) / 8
            positive = probabilities[labels == 1, None]
            negative = probabilities[None, labels == 0]
            expected = np.mean((positive > negative) + 0.5 * (positive == negative))
            self.assertAlmostEqual(metrics(probabilities, labels)["auc"], expected, places=14)

    def test_perfect_and_inverted_predictions(self):
        perfect = metrics([0, 1, 1, 0], [0, 1, 1, 0])
        self.assertEqual(perfect["auc"], 1.0)
        self.assertEqual(perfect["acc"], 1.0)
        self.assertEqual(perfect["brier"], 0.0)
        self.assertEqual(perfect["ece15"], 0.0)
        self.assertAlmostEqual(perfect["nll"], -np.log(1 - 1e-7), places=12)
        inverted = metrics([1, 0, 0, 1], [0, 1, 1, 0])
        self.assertEqual(inverted["auc"], 0.0)
        self.assertEqual(inverted["acc"], 0.0)
        self.assertEqual(inverted["brier"], 1.0)
        self.assertEqual(inverted["ece15"], 1.0)

    def test_single_class_has_no_auc(self):
        self.assertIsNone(metrics([0.2, 0.8], [1, 1])["auc"])

    def test_eval_overlapping_windows_keeps_exact_target_coverage(self):
        rows = [((0, 1, 2, 0, 2, 1, 1), (0, 1, 0, 0, 1, 1, 0)),
                ((2, 0), (1, 0))]
        dataset = WindowDataset(rows, [1, 0], sequence_length=4, step=2)
        expected = dataset.expected_targets()
        question = torch.tensor([[0.0], [0.1], [0.8], [0.4]])
        result, predictions = evaluate(ItemProbability(), loader(dataset, 2), question,
                                       torch.zeros_like(question), expected, torch.device("cpu"))
        expected_p = question[torch.as_tensor(expected["qid"]) + 1, 0].numpy()
        np.testing.assert_array_equal(predictions, expected_p)
        self.assertEqual(result["count"], 9)
        self.assertEqual(result, metrics(expected_p, expected["labels"]))
        wrong = {**expected, "labels": 1 - expected["labels"]}
        with self.assertRaisesRegex(AssertionError, "ordering/count"):
            evaluate(ItemProbability(), loader(dataset, 2), question,
                     torch.zeros_like(question), wrong, torch.device("cpu"))

    def test_batch_trimming_preserves_mask_and_prefix(self):
        qids = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        responses = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        mask = qids != 0
        trimmed = batch_to_device((qids, responses, mask), torch.device("cpu"))
        self.assertEqual(trimmed[0].shape, (2, 2))
        torch.testing.assert_close(trimmed[0], qids[:, :2])
        torch.testing.assert_close(trimmed[1], responses[:, :2])
        torch.testing.assert_close(trimmed[2], mask[:, :2])
        self.assertEqual(int(trimmed[2].sum()), 3)


if __name__ == "__main__":
    unittest.main()
