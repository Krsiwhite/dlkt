"""Meaningful protocol tests: coverage, cold-label exclusion, and frozen input."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.data import ColdQueryDataset, WindowDataset, prepare, split_students


class DataProtocolTests(unittest.TestCase):
    def test_preserved_student_split_counts_and_disjointness(self):
        split = split_students(4918, 42)
        self.assertEqual([len(split[key]) for key in ("train", "val", "test")], [2950, 492, 1476])
        self.assertEqual(len(set().union(*(set(ids) for ids in split.values()))), 4918)
        self.assertEqual(sum(map(len, split.values())), 4918)
        self.assertEqual(split, split_students(4918, 42))

    def test_window_coverage_including_tails_and_empty_filtered_students(self):
        for length in (1, 4, 5, 6, 8, 9, 10, 11, 21):
            rows = [(tuple(range(length)), tuple(i % 2 for i in range(length))), ((), ())]
            for step in (1, 2, 5):
                dataset = WindowDataset(rows, [0, 1], sequence_length=5, step=step)
                qids, labels = [], []
                for index in range(len(dataset)):
                    q, y, mask = dataset[index]
                    self.assertTrue(torch.all(q[mask] > 0))
                    qids.extend((q[mask] - 1).tolist())
                    labels.extend(y[mask].int().tolist())
                self.assertEqual(qids, list(range(length)))
                self.assertEqual(labels, list(rows[0][1]))
                self.assertEqual(qids, dataset.expected_targets()["qid"].tolist())

    def test_cold_history_contains_only_preceding_train_responses(self):
        rows = [((3, 0, 4, 1, 3, 2, 4, 0, 3), (1, 0, 1, 1, 0, 0, 0, 1, 1))]
        dataset = ColdQueryDataset(rows, [0], [0, 1, 2], [3], sequence_length=3)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.expected_targets()["interaction_index"].tolist(), [0, 4, 8])
        for i in range(len(dataset)):
            qids, responses, mask = dataset[i]
            query_position = int(mask.nonzero().item())
            _, end, qid, label, raw_position = dataset.queries[i]
            expected = [(q, y) for q, y in zip(rows[0][0][:raw_position], rows[0][1][:raw_position])
                        if q in {0, 1, 2}][-2:]
            self.assertEqual((qids[:query_position] - 1).tolist(), [q for q, _ in expected])
            self.assertEqual(responses[:query_position].int().tolist(), [y for _, y in expected])
            self.assertEqual(int(qids[query_position]), qid + 1)
            self.assertEqual(int(responses[query_position]), label)
            self.assertEqual(int(mask.sum()), 1)
            self.assertTrue(torch.all(qids[query_position + 1:] == 0))
        # Changing ANY cold response cannot change ANY query's observed history.
        changed = [(rows[0][0], tuple(1-y if q in {3, 4} else y
                                     for q, y in zip(*rows[0])))]
        other = ColdQueryDataset(changed, [0], [0, 1, 2], [3], sequence_length=3)
        for i in range(len(dataset)):
            q, y, mask = dataset[i]
            q2, y2, mask2 = other[i]
            self.assertTrue(torch.equal(q, q2))
            self.assertTrue(torch.equal(mask, mask2))
            self.assertTrue(torch.equal(y[~mask], y2[~mask2]))

    def test_prepare_teacher_training_only_and_unchanged_semantic_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [((tuple(range(10))), tuple((i + student) % 2 for i in range(10)))
                    for student in range(20)]
            text = "\n".join(part for q, y in rows
                             for part in (str(len(q)), ",".join(map(str, q)), ",".join(map(str, y)))) + "\n"
            (root / "used.txt").write_text(text, encoding="utf-8")
            embeddings = [{"QuestionId": i, "TextEmbedding": [i / 10, -i / 10],
                           "SubjectEmbedding": [0.1, 0.2]} for i in range(10)]
            (root / "used.json").write_text(json.dumps(embeddings), encoding="utf-8")
            for protocol in ("warm", "cold"):
                bundle = prepare(root, protocol, sequence_length=5, eval_step=2)
                self.assertFalse(bundle["question_table"].requires_grad)
                self.assertFalse(bundle["concept_table"].requires_grad)
                self.assertTrue(torch.equal(bundle["question_table"][1:],
                                            torch.tensor([e["TextEmbedding"] for e in embeddings])))
                self.assertTrue(set(bundle["train_student_index"]).issubset(bundle["split"]["train"]))
                self.assertTrue(set(bundle["train_question_ids"]).issubset(bundle["train_item_ids"]))
                for partition in ("train", "val", "test"):
                    emitted_q, emitted_y = [], []
                    dataset = bundle[partition]
                    for i in range(len(dataset)):
                        q, y, mask = dataset[i]
                        emitted_q.extend((q[mask] - 1).tolist())
                        emitted_y.extend(y[mask].int().tolist())
                    np.testing.assert_array_equal(emitted_q, bundle["expected"][partition]["qid"])
                    np.testing.assert_array_equal(emitted_y, bundle["expected"][partition]["labels"])
                if protocol == "cold":
                    self.assertTrue(set(bundle["train_question_ids"]).isdisjoint(bundle["expected"]["val"]["qid"]))
                    self.assertTrue(set(bundle["train_question_ids"]).isdisjoint(bundle["expected"]["test"]["qid"]))
                    self.assertEqual(bundle["metadata"]["item_partition_counts"], {"train": 7, "val": 1, "test": 2})


if __name__ == "__main__":
    unittest.main()
