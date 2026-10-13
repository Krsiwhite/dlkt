"""Checks for the architectural claims, not just output shapes."""

import unittest

import torch
from torch import nn

from experiments.models import build_model


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314)
        torch.set_num_threads(2)
        self.c = torch.randn(2, 7, 256)
        self.q = torch.randn(2, 7, 256)
        self.r = torch.randint(0, 2, (2, 7, 1)).float()

    def model(self, variant="decoupled"):
        return build_model(variant, model_dim=32, seq_len=10, dropout=0.2).eval()

    def test_probabilities_shapes_and_parameter_ranges(self):
        for variant in ("original", "semantic_mlp", "decoupled", "aligned"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                with torch.no_grad():
                    p = model(self.c, self.q, self.r)
                self.assertEqual(p.shape, (2, 7, 1))
                self.assertTrue(torch.isfinite(p).all())
                self.assertTrue(((p >= 0) & (p <= 1)).all())
        params = self.model().item_parameters(self.c, self.q)
        self.assertTrue(torch.isfinite(params["a"]).all())
        self.assertTrue((params["a"] >= 0.1).all())
        self.assertTrue((params["b"].abs() <= 3.0).all())

    def test_current_and_future_responses_cannot_affect_current_prediction(self):
        # One mutation at t must leave every prediction through t unchanged.
        for variant in ("original", "semantic_mlp", "decoupled"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                with torch.no_grad():
                    reference = model(self.c, self.q, self.r)
                    changed = self.r.clone()
                    changed[:, 3:] = 1.0 - changed[:, 3:]
                    perturbed = model(self.c, self.q, changed)
                torch.testing.assert_close(reference[:, :4], perturbed[:, :4],
                                           rtol=0, atol=1e-6)

    def test_future_questions_cannot_affect_earlier_predictions(self):
        for variant in ("original", "semantic_mlp", "decoupled"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                with torch.no_grad():
                    reference = model(self.c, self.q, self.r)
                    changed_c, changed_q = self.c.clone(), self.q.clone()
                    changed_c[:, 4:] = torch.randn_like(changed_c[:, 4:])
                    changed_q[:, 4:] = torch.randn_like(changed_q[:, 4:])
                    perturbed = model(changed_c, changed_q, self.r)
                torch.testing.assert_close(reference[:, :4], perturbed[:, :4],
                                           rtol=0, atol=1e-6)

    def test_current_question_does_not_change_current_ability(self):
        model = self.model()
        changed_c, changed_q = self.c.clone(), self.q.clone()
        changed_c[:, 3] = torch.randn_like(changed_c[:, 3])
        changed_q[:, 3] = torch.randn_like(changed_q[:, 3])
        with torch.no_grad():
            reference = model.ability(self.c, self.q, self.r)
            perturbed = model.ability(changed_c, changed_q, self.r)
        torch.testing.assert_close(reference[:, :4], perturbed[:, :4], rtol=0, atol=0)
        # This is not a history-blind model: the changed item may affect t+1.
        self.assertGreater((reference[:, 4:] - perturbed[:, 4:]).abs().max().item(), 1e-7)

    def test_item_parameters_invariant_to_history_position_and_training_mode(self):
        model = self.model().train()
        with torch.no_grad():
            reference = model.item_parameters(self.c[:, 2], self.q[:, 2])
            permutation = torch.tensor([6, 4, 2, 1, 0, 5, 3])
            reordered = model.item_parameters(self.c[:, permutation], self.q[:, permutation])
            repeated_c = self.c[:, 2:3].expand(-1, 7, -1)
            repeated_q = self.q[:, 2:3].expand(-1, 7, -1)
            repeated = model.item_parameters(repeated_c, repeated_q)
            again = model.item_parameters(repeated_c, repeated_q)
        for key in ("a", "b"):
            torch.testing.assert_close(reference[key], reordered[key][:, 2], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(repeated[key], reference[key][:, None].expand(-1, 7, -1),
                                       rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(repeated[key], again[key], rtol=0, atol=0)

    def test_raw_vectors_unchanged_and_receive_no_gradient(self):
        model = self.model().train()
        c, q = self.c.clone().requires_grad_(), self.q.clone().requires_grad_()
        p = model(c, q, self.r)
        loss = nn.functional.binary_cross_entropy(p, self.r)
        loss = loss + 0.001 * model.regularization(c[:, 0], q[:, 0])
        loss.backward()
        self.assertIsNone(c.grad)
        self.assertIsNone(q.grad)
        torch.testing.assert_close(c, self.c)
        torch.testing.assert_close(q, self.q)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in model.modules()))

    def test_padding_does_not_change_real_prefix(self):
        for variant in ("original", "semantic_mlp", "decoupled"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                with torch.no_grad():
                    full = model(self.c, self.q, self.r)
                    short = model(self.c[:, :4], self.q[:, :4], self.r[:, :4])
                torch.testing.assert_close(full[:, :4], short, rtol=1e-5, atol=1e-6)

    def test_alignment_variant_is_exactly_same_architecture(self):
        torch.manual_seed(8)
        base = self.model("decoupled")
        torch.manual_seed(8)
        aligned = self.model("aligned")
        self.assertEqual(base.state_dict().keys(), aligned.state_dict().keys())
        for key in base.state_dict():
            torch.testing.assert_close(base.state_dict()[key], aligned.state_dict()[key],
                                       rtol=0, atol=0)

    def test_first_ability_is_shared_prior_and_response_has_later_effect(self):
        model = self.model()
        with torch.no_grad():
            first = model.ability(self.c, self.q, self.r)
            second = model.ability(self.c, self.q, 1.0 - self.r)
        torch.testing.assert_close(first[:, 0], first[0:1, 0].expand(2, -1), rtol=0, atol=0)
        torch.testing.assert_close(first[:, 0], second[:, 0], rtol=0, atol=0)
        self.assertGreater((first[:, 1:] - second[:, 1:]).abs().max().item(), 1e-7)


if __name__ == "__main__":
    unittest.main()
