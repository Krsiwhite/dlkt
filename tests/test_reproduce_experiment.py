"""CPU toy coverage of the standalone checkpoint verification path."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.data import prepare
from experiments.models import build_model
from experiments.run import evaluate, loader, seed_all, source_hashes
from tools.reproduce_experiment import reproduce


def dump(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="dlkt-reproduction-test-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.run = root / "completed-run"
        self.data = root / "data"
        self.run.mkdir()
        self.data.mkdir()
        rng = np.random.default_rng(83)
        embeddings = [{"QuestionId": qid, "TextEmbedding": rng.normal(size=256).tolist(),
                       "SubjectEmbedding": rng.normal(size=256).tolist()} for qid in range(8)]
        dump(self.data / "used.json", embeddings)
        lines = []
        for student in range(20):
            lines.extend(["6", ",".join(str((student + i) % 8) for i in range(6)),
                          "0,1,0,1,1,0"])
        (self.data / "used.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        bundle = prepare(self.data, "warm", sequence_length=4, eval_step=2)
        self.config = {
            "protocol": "warm", "variant": "decoupled", "seed": 42,
            "split_seed": 42, "item_seed": 20261013, "sequence_length": 4,
            "eval_step": 2, "batch_size": 3, "model_dim": 16,
            "new_model_dropout": 0.1, "smoke": False,
            "source_sha256": source_hashes(), "data_sha256": bundle["metadata"]["source_sha256"],
            "protocol_sha256": hashlib.sha256(
                json.dumps(bundle["metadata"], sort_keys=True).encode()).hexdigest(),
        }
        torch.set_num_threads(6)
        seed_all(42)
        model = build_model("decoupled", model_dim=16, seq_len=4)
        score, p = evaluate(model, loader(bundle["test"], 3), bundle["question_table"],
                            bundle["concept_table"], bundle["expected"]["test"], torch.device("cpu"))
        validation = {"auc": 0.5}
        torch.save({"model": model.state_dict(), "config": self.config,
                    "epoch": 1, "validation": validation}, self.run / "best.pt")
        self.summary = {"config": self.config, "best_epoch": 1, "validation": validation,
                        "test": score, "checkpoint_sha256": digest(self.run / "best.pt")}
        dump(self.run / "config.json", self.config)
        dump(self.run / "summary.json", self.summary)
        np.savez_compressed(self.run / "test_predictions.npz", probabilities=p,
                            **bundle["expected"]["test"])

    def test_complete_cpu_reproduction_is_exact_and_does_not_modify_artifacts(self):
        before = {path.name: digest(path) for path in self.run.iterdir()}
        report = reproduce(self.run, self.data, device="cpu")
        self.assertTrue(report["predictions"]["exact_equal"])
        self.assertTrue(report["predictions"]["bitwise_equal"])
        self.assertEqual(report["predictions"]["max_absolute_difference"], 0.0)
        self.assertEqual(report["reference_metrics"], report["reproduced_metrics"])
        self.assertTrue((self.run / "reproduction.json").is_file())
        self.assertEqual(before, {name: digest(self.run / name) for name in before})
        with self.assertRaises(FileExistsError):
            reproduce(self.run, self.data, device="cpu")

    def test_changed_dataset_is_refused_before_inference(self):
        with (self.data / "used.txt").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Input used.txt SHA-256"):
            reproduce(self.run, self.data, device="cpu")
        self.assertFalse((self.run / "reproduction.json").exists())

    def test_smoke_and_unfinished_runs_are_refused(self):
        smoke = {**self.config, "smoke": True}
        dump(self.run / "config.json", smoke)
        with self.assertRaisesRegex(ValueError, "Smoke or unfinished"):
            reproduce(self.run, self.data, device="cpu")
        dump(self.run / "config.json", self.config)
        dump(self.run / "summary.json", {**self.summary, "test": None})
        with self.assertRaisesRegex(ValueError, "Smoke or unfinished"):
            reproduce(self.run, self.data, device="cpu")

    def test_changed_source_hash_is_refused(self):
        self.config["source_sha256"] = {"experiments/models.py": "0" * 64}
        dump(self.run / "config.json", self.config)
        self.summary["config"] = self.config
        dump(self.run / "summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "Training source hashes"):
            reproduce(self.run, self.data, device="cpu")

    def test_changed_checkpoint_is_refused(self):
        with (self.run / "best.pt").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256"):
            reproduce(self.run, self.data, device="cpu")


if __name__ == "__main__":
    unittest.main()
