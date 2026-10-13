"""Synthetic integrity checks for the frozen Transformer report and inference tools."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import summarize_transformer as report


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class TransformerReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="transformer-report-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite, self.data = self.root / "suite", self.root / "data"
        self.suite.mkdir()
        self.data.mkdir()
        (self.data / "used.txt").write_text("synthetic fixture", encoding="utf-8")
        (self.data / "used.json").write_text("[]", encoding="utf-8")
        data_hashes = {name: report.file_sha256(self.data / name) for name in ("used.txt", "used.json")}
        sources = report.current_source_hashes()
        self.arrays = {"student_index": np.array([0, 0, 1, 1]),
                       "interaction_index": np.array([0, 1, 0, 1]),
                       "qid": np.array([1, 2, 1, 3]), "labels": np.array([0, 0, 1, 1])}
        self.p = np.array([0., .5, .5, 1.], dtype=np.float32)
        metric = report.checked_metrics(self.p, self.arrays["labels"])
        protocols = {}
        for protocol in report.PROTOCOLS:
            metadata = {"protocol": protocol, "source_sha256": data_hashes,
                        "partitions": {"test": {"target_order_sha256": report.identity_sha256(self.arrays), "targets": 4}}}
            protocols[protocol] = metadata
            dump(self.suite / f"{protocol}-protocol.json", metadata)
        for (protocol, variant, seed), directory in report.planned_runs(self.suite).items():
            directory.mkdir()
            config = {"protocol": protocol, "variant": variant, "seed": seed, "smoke": False,
                      "backbone": "original_TextAKT3PL_2blocks_8heads", "epochs": 40,
                      "anchor_weight": 0, "alignment_weight": .05 if variant == "aligned" else 0,
                      "teacher_metadata": {"fit": "train_only"} if variant == "aligned" else None,
                      "source_sha256": sources, "data_sha256": data_hashes,
                      "protocol_sha256": report.protocol_sha256(protocols[protocol])}
            # Hash verification needs bytes, never executable pickle fixtures.
            (directory / "best.pt").write_bytes(b"checkpoint fixture")
            summary = {"config": config, "test": metric, "validation": metric, "best_epoch": 1,
                       "epochs_run": 2, "seconds": 1., "trainable_parameters": 123,
                       "checkpoint_sha256": report.file_sha256(directory / "best.pt")}
            dump(directory / "config.json", config)
            dump(directory / "summary.json", summary)
            np.savez_compressed(directory / "test_predictions.npz", probabilities=self.p, **self.arrays)
        self.one = self.suite / "cold-aligned-s3407"

    def test_complete_report_recomputes_all_metrics_and_three_contrasts(self):
        result = report.aggregate(self.suite, self.data)
        self.assertTrue(result["completion_status"]["all_planned_18_runs_audited"])
        self.assertEqual(len(result["runs"]), 18)
        self.assertEqual(len(result["groups"]), 6)
        self.assertEqual(len(result["paired_comparisons"]), 6)
        self.assertEqual(result["groups"][0]["test"]["auc"]["mean"], .875)
        self.assertEqual(result["groups"][0]["test"]["acc"]["mean"], .75)
        for contrast in result["paired_comparisons"]:
            self.assertEqual(contrast["delta_auc_percentage_points"]["mean"], 0)
            self.assertEqual(set(contrast["delta_metrics"]), set(report.METRICS))
        self.assertIn("18", report.markdown(result))

    def test_incomplete_suite_does_not_open_any_prediction_archive(self):
        (self.one / "summary.json").unlink()
        with patch.object(report.np, "load", side_effect=AssertionError("Prediction must not be opened")):
            self.assertEqual(report.completion_status(self.suite)["completed_artifact_sets"], 17)
            with self.assertRaisesRegex(ValueError, "Incomplete planned suite"):
                report.aggregate(self.suite, self.data)

    def test_non_auc_metric_corruption_is_rejected(self):
        path = self.one / "summary.json"
        summary = report.read_json(path)
        summary["test"]["nll"] += .01
        dump(path, summary)
        with self.assertRaisesRegex(ValueError, "recomputed nll differs"):
            report.aggregate(self.suite, self.data)

    def test_cross_seed_target_reordering_is_rejected(self):
        changed = {key: value.copy() for key, value in self.arrays.items()}
        changed["student_index"][[0, 2]] = changed["student_index"][[2, 0]]
        np.savez_compressed(self.one / "test_predictions.npz", probabilities=self.p, **changed)
        with self.assertRaisesRegex(ValueError, "Test identities/order changed"):
            report.aggregate(self.suite, self.data)

    def test_checkpoint_corruption_is_rejected_before_predictions(self):
        (self.one / "best.pt").write_bytes(b"different weights")
        with patch.object(report.np, "load", side_effect=AssertionError("Prediction must not be opened")):
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256 differs"):
                report.aggregate(self.suite, self.data)

    def test_current_data_mutation_is_rejected(self):
        (self.data / "used.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Current input hashes differ"):
            report.aggregate(self.suite, self.data)

    def test_current_source_mutation_is_rejected(self):
        with patch.object(report, "current_source_hashes", return_value={"changed.py": "a" * 64}):
            with self.assertRaisesRegex(ValueError, "Current source hashes differ"):
                report.aggregate(self.suite, self.data)

    def test_common_recipe_drift_is_rejected(self):
        config = report.read_json(self.one / "config.json")
        summary = report.read_json(self.one / "summary.json")
        config["epochs"] = 41
        summary["config"] = config
        dump(self.one / "config.json", config)
        dump(self.one / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "Common training settings changed"):
            report.aggregate(self.suite, self.data)

    def test_metrics_reject_invalid_probability_and_labels(self):
        with self.assertRaisesRegex(ValueError, "Invalid probabilities"):
            report.checked_metrics(np.array([1.01, .2]), np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, "Invalid probabilities"):
            report.checked_metrics(np.array([.1, .2]), np.array([0, .5]))


class TransformerReproductionTests(unittest.TestCase):
    def test_saved_static_transformer_cpu_inference_is_exact(self):
        # No optimizer or teacher fitting: this validates construction/loading
        # of a saved Transformer through the standalone verification entrypoint.
        import hashlib
        import torch
        from experiments.data import prepare
        from transformer_experiments.models import build_model
        from transformer_experiments.run import evaluate, loader, seed_all, source_hashes
        from tools.reproduce_transformer import reproduce

        with tempfile.TemporaryDirectory(prefix="transformer-reproduce-test-") as temporary:
            root = Path(temporary)
            run, data = root / "run", root / "data"
            run.mkdir()
            data.mkdir()
            rng = np.random.default_rng(83)
            dump(data / "used.json", [{"QuestionId": qid, "TextEmbedding": rng.normal(size=256).tolist(),
                                      "SubjectEmbedding": rng.normal(size=256).tolist()} for qid in range(8)])
            lines = []
            for student in range(20):
                lines.extend(["6", ",".join(str((student + i) % 8) for i in range(6)), "0,1,0,1,1,0"])
            (data / "used.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            bundle = prepare(data, "warm", sequence_length=4, eval_step=2)
            config = {"protocol": "warm", "variant": "static", "seed": 42,
                      "split_seed": 42, "item_seed": 20261013, "sequence_length": 4,
                      "eval_step": 2, "batch_size": 3, "model_dim": 16, "smoke": False,
                      "source_sha256": source_hashes(), "data_sha256": bundle["metadata"]["source_sha256"],
                      "protocol_sha256": hashlib.sha256(json.dumps(bundle["metadata"], sort_keys=True).encode()).hexdigest()}
            torch.set_num_threads(2)
            seed_all(42)
            model = build_model("static", model_dim=16, seq_len=4)
            metric, probabilities = evaluate(model, loader(bundle["test"], 3), bundle["question_table"],
                                             bundle["concept_table"], bundle["expected"]["test"], torch.device("cpu"))
            torch.save({"model": model.state_dict(), "config": config, "epoch": 1, "validation": metric}, run / "best.pt")
            dump(run / "config.json", config)
            dump(run / "summary.json", {"config": config, "best_epoch": 1, "validation": metric, "test": metric,
                                        "checkpoint_sha256": report.file_sha256(run / "best.pt")})
            np.savez_compressed(run / "test_predictions.npz", probabilities=probabilities, **bundle["expected"]["test"])
            before = {path.name: report.file_sha256(path) for path in run.iterdir()}
            result = reproduce(run, data, "cpu")
            self.assertTrue(result["predictions"]["bitwise_equal"])
            self.assertEqual(result["metric_differences"], {key: 0 for key in report.METRICS})
            self.assertEqual(before, {name: report.file_sha256(run / name) for name in before})
            with self.assertRaises(FileExistsError):
                reproduce(run, data, "cpu")


if __name__ == "__main__":
    unittest.main()
