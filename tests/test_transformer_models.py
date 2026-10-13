"""Tests of the controlled changes and causality on the actual original AKT."""

import unittest

import torch
from torch import nn

from experiments.models import build_model as old_build_model
from experiments.teacher import alignment_loss
from model.my_model import Architecture, TextAKT3PL, TransformerLayer
from transformer_experiments.models import build_model


class TransformerModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(314)
        self.c = torch.randn(2, 7, 256)
        self.q = torch.randn(2, 7, 256)
        self.r = torch.randint(0, 2, (2, 7, 1)).float()

    def model(self, variant="static"):
        return build_model(variant, model_dim=32, seq_len=10).eval()

    def test_original_matches_existing_wrapper_exactly(self):
        torch.manual_seed(21)
        old = old_build_model("original", model_dim=32, seq_len=10).eval()
        torch.manual_seed(21)
        new = self.model("original")
        self.assertEqual(old.state_dict().keys(), new.state_dict().keys())
        for key in old.state_dict():
            torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key],
                                       rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(old(self.c, self.q, self.r),
                                       new(self.c, self.q, self.r), rtol=0, atol=0)

    def test_initial_parameters_rng_and_akt_modules_are_identical(self):
        models, rngs = [], []
        for variant in ("original", "static", "aligned"):
            torch.manual_seed(22)
            model = self.model(variant)
            models.append(model)
            rngs.append(torch.get_rng_state().clone())
            self.assertIs(type(model.model), TextAKT3PL)
            self.assertIs(type(model.model.akt_encoder), Architecture)
            self.assertIs(type(model.model.extractor.mix_diff), TransformerLayer)
            self.assertFalse(any(isinstance(module, nn.GRU) for module in model.modules()))
            self.assertEqual(len(model.model.akt_encoder.question_encoder), 2)
            self.assertEqual(len(model.model.akt_encoder.knowledge_encoder), 2)
            self.assertEqual(len(model.model.akt_encoder.retriever), 2)
            # The only lookup tables are the original two position embeddings.
            embeddings = [name for name, module in model.named_modules()
                          if isinstance(module, nn.Embedding)]
            self.assertEqual(embeddings, ["model.extractor.pos_encoder.pe",
                                          "model.pos_encoder.pe"])
        for model, rng in zip(models[1:], rngs[1:]):
            self.assertEqual(models[0].state_dict().keys(), model.state_dict().keys())
            for key in model.state_dict():
                torch.testing.assert_close(models[0].state_dict()[key], model.state_dict()[key],
                                           rtol=0, atol=0)
            torch.testing.assert_close(rngs[0], rng, rtol=0, atol=0)
        torch.manual_seed(22)
        default_model = build_model("static")
        self.assertEqual(sum(p.numel() for p in default_model.parameters()), 2591164)

    def test_static_retains_identical_contextual_features(self):
        torch.manual_seed(23)
        original = self.model("original")
        torch.manual_seed(23)
        static = self.model("static")
        # Forward first synchronizes the legacy module-global mask device.
        with torch.no_grad():
            original(self.c, self.q, self.r)
            old = original.model.extractor(self.c, self.q)
            new = static.model.extractor(self.c, self.q)
        for index in (0, 1):
            torch.testing.assert_close(old[index], new[index], rtol=0, atol=0)
        self.assertGreater((old[2] - new[2]).abs().max().item(), 1e-6)

    def test_current_and_future_responses_do_not_leak(self):
        for variant in ("original", "static", "aligned"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                with torch.no_grad():
                    baseline = model(self.c, self.q, self.r)
                    for position in (0, 3, 6):
                        changed = self.r.clone()
                        changed[:, position:] = 1.0 - changed[:, position:]
                        actual = model(self.c, self.q, changed)
                        torch.testing.assert_close(baseline[:, :position + 1],
                                                   actual[:, :position + 1],
                                                   rtol=0, atol=1e-6)
                    all_flipped = model(self.c, self.q, 1.0 - self.r)
                    self.assertGreater((baseline[:, 1:] - all_flipped[:, 1:]).abs().max().item(),
                                       1e-7)

    def test_future_questions_and_prefix_extension_do_not_leak(self):
        for variant in ("original", "static", "aligned"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                changed_c, changed_q = self.c.clone(), self.q.clone()
                changed_c[:, 4:] = torch.randn_like(changed_c[:, 4:])
                changed_q[:, 4:] = torch.randn_like(changed_q[:, 4:])
                with torch.no_grad():
                    baseline = model(self.c, self.q, self.r)
                    changed = model(changed_c, changed_q, self.r)
                    short = model(self.c[:, :4], self.q[:, :4], self.r[:, :4])
                torch.testing.assert_close(baseline[:, :4], changed[:, :4], rtol=0, atol=1e-6)
                torch.testing.assert_close(baseline[:, :4], short, rtol=1e-5, atol=1e-6)

    def test_item_heads_are_invariant_in_train_and_eval(self):
        model = self.model()
        for train_mode in (False, True):
            with self.subTest(train_mode=train_mode):
                model.train(train_mode)
                with torch.no_grad():
                    reference = model.item_parameters(self.c[:, 2], self.q[:, 2])
                    c = self.c[:, 2:3].expand(-1, 7, -1)
                    q = self.q[:, 2:3].expand(-1, 7, -1)
                    repeated = model.item_parameters(c, q)
                    again = model.item_parameters(c, q)
                    # These are the actual forward-path head values, including
                    # random contextual dropout in training; they must match.
                    model(self.c, self.q, self.r)
                    extracted = model.model.extractor(c, q)
                for name, index in (("b", 2), ("a", 3), ("guess", 4)):
                    expected = reference[name][:, None].expand(-1, 7, -1)
                    torch.testing.assert_close(repeated[name], expected, rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(repeated[name], again[name], rtol=0, atol=0)
                    torch.testing.assert_close(repeated[name], extracted[index], rtol=0, atol=0)
                self.assertTrue((repeated["b"].abs() <= 3).all())
                self.assertTrue((repeated["a"] >= 0.1).all())
                self.assertTrue(((repeated["guess"] >= 0) & (repeated["guess"] <= 1)).all())

    def test_alignment_uses_shared_head_and_detaches_raw_inputs(self):
        model = self.model("aligned").train()
        c = self.c[0].clone().requires_grad_()
        q = self.q[0].clone().requires_grad_()
        target = torch.linspace(-1, 1, 6, requires_grad=True)
        teacher = {"item_ids": torch.arange(6), "difficulty": target,
                   "weights": torch.full((6,), 0.8)}
        loss = alignment_loss(model, c, q, teacher)
        loss.backward()
        self.assertIsNone(c.grad)
        self.assertIsNone(q.grad)
        self.assertIsNone(target.grad)
        for module in (model.model.extractor.concept_proj,
                       model.model.extractor.q_shared,
                       model.model.extractor.head_mu):
            for p in module.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIsNone(model.model.extractor.pos_encoder.pe.weight.grad)
        self.assertIsNone(model.model.extractor.mix_diff.linear1.weight.grad)
        self.assertGreater(model.model.extractor.head_mu[-1].weight.grad.abs().sum().item(), 0)

    def test_full_forward_backward_preserves_raw_vectors(self):
        for variant in ("original", "static", "aligned"):
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                c, q = self.c.clone().requires_grad_(), self.q.clone().requires_grad_()
                response = self.r.clone().requires_grad_()
                probabilities = model(c, q, response)
                self.assertTrue(torch.isfinite(probabilities).all())
                nn.functional.binary_cross_entropy(probabilities, self.r).backward()
                self.assertIsNone(c.grad)
                self.assertIsNone(q.grad)
                self.assertIsNone(response.grad)
                torch.testing.assert_close(c, self.c, rtol=0, atol=0)
                torch.testing.assert_close(q, self.q, rtol=0, atol=0)
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_static_and_aligned_forward_match_and_checkpoint_reload(self):
        torch.manual_seed(24)
        static = self.model("static")
        torch.manual_seed(24)
        aligned = self.model("aligned")
        with torch.no_grad():
            expected = static(self.c, self.q, self.r)
            torch.testing.assert_close(expected, aligned(self.c, self.q, self.r), rtol=0, atol=0)
            reloaded = self.model("aligned")
            reloaded.load_state_dict(static.state_dict(), strict=True)
            torch.testing.assert_close(expected, reloaded(self.c, self.q, self.r), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
