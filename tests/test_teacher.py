"""Checks for training-only supervision, item ordering and semantic gradients."""

import math
import unittest

import torch

from experiments.teacher import alignment_loss, fit_teacher


class SemanticHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 1)

    def item_parameters(self, concepts, questions):
        return {"b": self.head(torch.cat((concepts, questions), dim=-1))}


class TeacherTests(unittest.TestCase):
    def test_synthetic_ordering_and_unseen_items(self):
        students, questions, labels = [], [], []
        abilities = [-1.5, -0.5, 0.5, 1.5] * 2
        for student, ability in enumerate(abilities):
            for question, difficulty in enumerate([-1.2, 0.0, 1.2]):
                correct = round(200 / (1 + math.exp(-(ability - difficulty))))
                students.extend([student * 17 + 3] * 200)
                questions.extend([question] * 200)
                labels.extend([1] * correct + [0] * (200 - correct))
        result = fit_teacher(students, questions, labels, n_items=4)
        b = result["difficulty"]
        self.assertTrue(bool((b[1:3] > b[:2]).all()))
        self.assertAlmostEqual(float(b[:3].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(b[:3].std(unbiased=False)), 1.0, places=6)
        self.assertTrue(bool(torch.isnan(b[3])))
        self.assertEqual(int(result["counts"][3]), 0)
        self.assertEqual(float(result["weights"][3]), 0)
        self.assertEqual(result["item_ids"].tolist(), [0, 1, 2])
        self.assertEqual(result["metadata"]["n_students"], 8)
        self.assertEqual(result["metadata"]["n_observations"], 4800)
        self.assertTrue(result["metadata"]["converged"])

    def test_only_supplied_records_count_and_no_teacher_gradients(self):
        result = fit_teacher([100, 100, 500, 500], [1, 1, 3, 3], [1, 0, 0, 0], n_items=5)
        self.assertEqual(result["counts"].tolist(), [0, 2, 0, 2, 0])
        self.assertEqual(result["item_ids"].tolist(), [1, 3])
        self.assertEqual(result["metadata"]["n_students"], 2)
        self.assertAlmostEqual(float(result["weights"][1]), 2 / 22, places=6)
        for name in ("difficulty", "raw_difficulty", "weights"):
            self.assertFalse(result[name].requires_grad)
        repeated = fit_teacher([100, 100, 500, 500], [1, 1, 3, 3], [1, 0, 0, 0], n_items=5)
        self.assertEqual(result["metadata"]["training_records_sha256"], repeated["metadata"]["training_records_sha256"])

    def test_alignment_backprop_and_no_unseen_supervision(self):
        result = fit_teacher([4, 4, 10, 10], [0, 2, 0, 2], [1, 0, 1, 0], n_items=3)
        torch.manual_seed(7)
        model = SemanticHead()
        concepts = torch.randn(4, 2)
        questions = torch.randn(4, 2)
        first_loss = alignment_loss(model, concepts, questions, result)
        self.assertTrue(bool(torch.isfinite(first_loss)))
        # Item 1 (lookup row 2) is absent from training and cannot affect alignment.
        concepts[2] = 1000
        questions[2] = -1000
        second_loss = alignment_loss(model, concepts, questions, result)
        self.assertEqual(float(first_loss.detach()), float(second_loss.detach()))
        second_loss.backward()
        self.assertTrue(bool(torch.isfinite(model.head.weight.grad).all()))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_invalid_inputs_fail_closed(self):
        for args in (
            ([], [], [], 3),
            ([0], [3], [1], 3),
            ([0], [-1], [1], 3),
            ([0], [0.5], [1], 3),
            ([0], [0], [0.5], 3),
            ([0, 1], [0], [1], 3),
        ):
            with self.assertRaises(ValueError):
                fit_teacher(*args)


if __name__ == "__main__":
    unittest.main()
