"""Frozen semantic inputs and reproducible warm/cold item evaluation.

The triplet reader and overlapping-window convention are adapted from the
preserved ``dlkt-best-806453/runtime.py`` inference package. No retained source,
embedding, checkpoint, or student record is modified by this module.

Question IDs are used only to index immutable semantic tables and to audit the
split. They must never be passed to a trainable item-embedding layer.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_students(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) % 3:
        raise ValueError("used.txt must contain three lines per student")
    rows = []
    for offset in range(0, len(lines), 3):
        length = int(lines[offset])
        questions = tuple(map(int, lines[offset + 1].split(",")))
        labels = tuple(map(int, lines[offset + 2].split(",")))
        if length < 1 or len(questions) != length or len(labels) != length:
            raise ValueError(f"Invalid sequence at student row {offset // 3}")
        if min(questions) < 0 or any(label not in (0, 1) for label in labels):
            raise ValueError(f"Invalid item or response at student row {offset // 3}")
        rows.append((questions, labels))
    return rows


def split_students(count, split_seed=42):
    """Match the preserved 60/10/30 student split exactly (2950/492/1476)."""
    indices = list(range(count))
    random.Random(split_seed).shuffle(indices)
    trainval_end = int(count * 0.7)
    train_end = int(trainval_end * (1 - 1 / 7))
    result = {"train": indices[:train_end],
              "val": indices[train_end:trainval_end],
              "test": indices[trainval_end:]}
    if any(not ids for ids in result.values()):
        raise ValueError("All three student partitions must be nonempty")
    flattened = [student for ids in result.values() for student in ids]
    assert len(set(flattened)) == count == len(flattened)
    return result


def split_items(count, item_seed=20261013):
    indices = list(range(count))
    random.Random(item_seed).shuffle(indices)
    train_end = int(count * 0.7)
    val_end = train_end + int(count * 0.15)
    result = {"train": sorted(indices[:train_end]),
              "val": sorted(indices[train_end:val_end]),
              "test": sorted(indices[val_end:])}
    if any(not ids for ids in result.values()):
        raise ValueError("All three item partitions must be nonempty")
    assert sum(len(ids) for ids in result.values()) == count
    assert len(set().union(*(set(ids) for ids in result.values()))) == count
    return result


def make_windows(rows, indices, sequence_length, step):
    """(student, start, first_score) windows, each target scored exactly once.

    Later windows retain L-step historical positions and score only new
    positions. The first window scores all its real positions. Padding is on
    the right, and no fake padded target is ever scored.
    """
    if sequence_length < 1 or not 1 <= step <= sequence_length:
        raise ValueError("Need length >= 1 and 1 <= step <= length")
    windows = []
    for student in indices:
        length = len(rows[student][0])
        start = 0
        next_target = 0
        while start < length:
            first_score = 0 if start == 0 else sequence_length - step
            stop = min(start + sequence_length, length)
            if start + first_score < stop:
                if start + first_score != next_target:
                    raise RuntimeError("Duplicated, skipped, or reordered window targets")
                windows.append((student, start, first_score))
                next_target = stop
            if start + sequence_length >= length:
                break
            start += step
        if next_target != length:
            raise RuntimeError("Window targets do not cover the student sequence")
    return windows


def _tensor_rows(rows):
    return [(torch.tensor(q, dtype=torch.long) + 1,
             torch.tensor(y, dtype=torch.float32)) for q, y in rows]


class WindowDataset(Dataset):
    def __init__(self, rows, indices, sequence_length, step, original_positions=None):
        self.rows = rows
        self.indices = list(indices)
        self.sequence_length = sequence_length
        self.step = step
        self.windows = make_windows(rows, indices, sequence_length, step)
        self._tensors = _tensor_rows(rows)
        self.original_positions = original_positions

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        student, start, first_score = self.windows[index]
        q, y = self._tensors[student]
        stop = min(start + self.sequence_length, len(q))
        valid = stop - start
        qids = torch.zeros(self.sequence_length, dtype=torch.long)
        responses = torch.zeros(self.sequence_length, dtype=torch.float32)
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        qids[:valid] = q[start:stop]
        responses[:valid] = y[start:stop]
        mask[first_score:valid] = True
        return qids, responses, mask

    def expected_targets(self):
        result = {key: [] for key in ("labels", "qid", "student_index", "interaction_index")}
        for student in self.indices:
            q, y = self.rows[student]
            result["labels"].extend(y)
            result["qid"].extend(q)
            result["student_index"].extend([student] * len(q))
            positions = (range(len(q)) if self.original_positions is None
                         else self.original_positions[student])
            result["interaction_index"].extend(positions)
        return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}


class ColdQueryDataset(Dataset):
    """Each cold query sees only preceding train-item responses.

    The query label is returned for loss/scoring at its own position. The model
    must be strictly causal in responses; no earlier cold-query label is ever
    present in any history. This is stricter than online adaptation to new
    items: every cold item remains at zero observed responses throughout eval.
    """

    def __init__(self, rows, indices, train_item_ids, query_item_ids, sequence_length):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.train_item_ids = frozenset(train_item_ids)
        self.query_item_ids = frozenset(query_item_ids)
        if self.train_item_ids & self.query_item_ids:
            raise ValueError("Cold query items overlap with train items")
        self.histories = {}
        # (student, number of preceding train interactions, qid, label, rawposition)
        self.queries = []
        for student in indices:
            history_q, history_y, history_positions = [], [], []
            qids, labels = rows[student]
            for position, (qid, label) in enumerate(zip(qids, labels)):
                if qid in self.train_item_ids:
                    history_q.append(qid)
                    history_y.append(label)
                    history_positions.append(position)
                elif qid in self.query_item_ids:
                    self.queries.append((student, len(history_q), qid, label, position))
            self.histories[student] = (
                torch.tensor(history_q, dtype=torch.long) + 1,
                torch.tensor(history_y, dtype=torch.float32),
                tuple(history_positions),
            )
        self._audit()

    def _audit(self):
        previous = None
        seen_students = set()
        for student, end, qid, label, raw_position in self.queries:
            _, _, positions = self.histories[student]
            assert qid in self.query_item_ids and qid not in self.train_item_ids
            assert label in (0, 1) and 0 <= end <= len(positions)
            assert end == 0 or positions[end - 1] < raw_position
            assert end == len(positions) or positions[end] > raw_position
            if previous is not None and previous[0] == student:
                assert raw_position > previous[1]
            else:
                assert student not in seen_students
                seen_students.add(student)
            previous = (student, raw_position)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        student, end, qid, label, _ = self.queries[index]
        q, y, _ = self.histories[student]
        start = max(0, end - (self.sequence_length - 1))
        count = end - start
        qids = torch.zeros(self.sequence_length, dtype=torch.long)
        responses = torch.zeros(self.sequence_length, dtype=torch.float32)
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        qids[:count] = q[start:end]
        responses[:count] = y[start:end]
        qids[count] = qid + 1
        responses[count] = label
        mask[count] = True
        return qids, responses, mask

    def expected_targets(self):
        # Empty query sets are rejected by prepare, but direct toy use is valid.
        columns = np.asarray(self.queries, dtype=np.int64).reshape(-1, 5)
        return {"student_index": columns[:, 0], "qid": columns[:, 2],
                "labels": columns[:, 3], "interaction_index": columns[:, 4]}


def _load_tables(data_root, rows):
    embeddings = json.loads((data_root / "used.json").read_text(encoding="utf-8"))
    if not embeddings or any(row["QuestionId"] != i for i, row in enumerate(embeddings)):
        raise ValueError("QuestionId rows must be sorted, contiguous, and zero based")
    dim = len(embeddings[0]["TextEmbedding"])
    if dim < 1 or any(len(row["TextEmbedding"]) != dim or
                      len(row["SubjectEmbedding"]) != dim for row in embeddings):
        raise ValueError("Inconsistent frozen semantic dimensions")
    if any(max(q) >= len(embeddings) for q, _ in rows):
        raise ValueError("Student sequence references missing semantic vectors")
    zero = [0.0] * dim
    question = torch.tensor([zero] + [r["TextEmbedding"] for r in embeddings], dtype=torch.float32)
    concept = torch.tensor([zero] + [r["SubjectEmbedding"] for r in embeddings], dtype=torch.float32)
    if not torch.isfinite(question).all() or not torch.isfinite(concept).all():
        raise ValueError("Semantic input contains nonfinite values")
    assert not question.requires_grad and not concept.requires_grad
    return question, concept


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_digest(targets):
    digest = hashlib.sha256()
    for key in ("student_index", "interaction_index", "qid", "labels"):
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(targets[key], dtype="<i8").tobytes())
    return digest.hexdigest()


def prepare(data_root, protocol, sequence_length=201, split_seed=42,
            item_seed=20261013, eval_step=100, train_step=None):
    """Build datasets, immutable tables, train-only teacher data, and audit data.

    Returns a plain dictionary. ``train``, ``val`` and ``test`` yield the tuple
    ``(qids, responses, score_mask)`` with shapes [L]. All qids are original IDs
    plus one; zero is padding. ``expected[partition]`` describes targets in
    DataLoader(shuffle=False) order. Teacher arrays use original zero-based IDs
    and global student indices, and contain training interactions only.
    """
    if protocol not in {"warm", "cold"}:
        raise ValueError("protocol must be warm or cold")
    data_root = Path(data_root)
    rows = read_students(data_root / "used.txt")
    split = split_students(len(rows), split_seed)
    question_table, concept_table = _load_tables(data_root, rows)
    item_count = question_table.shape[0] - 1
    input_dim = question_table.shape[1]
    train_step = sequence_length if train_step is None else train_step
    item_split = (split_items(item_count, item_seed) if protocol == "cold" else
                  {key: list(range(item_count)) for key in ("train", "val", "test")})
    train_items = frozenset(item_split["train"])
    if protocol == "cold":
        protocol_rows, positions = [], []
        for qids, labels in rows:
            keep = [i for i, qid in enumerate(qids) if qid in train_items]
            protocol_rows.append((tuple(qids[i] for i in keep), tuple(labels[i] for i in keep)))
            positions.append(tuple(keep))
    else:
        protocol_rows, positions = rows, None
    datasets = {"train": WindowDataset(protocol_rows, split["train"], sequence_length,
                                        train_step, positions)}
    for partition in ("val", "test"):
        if protocol == "warm":
            datasets[partition] = WindowDataset(rows, split[partition], sequence_length, eval_step)
        else:
            datasets[partition] = ColdQueryDataset(rows, split[partition], train_items,
                                                   item_split[partition], sequence_length)
    expected = {key: dataset.expected_targets() for key, dataset in datasets.items()}
    if any(len(target["labels"]) == 0 for target in expected.values()):
        raise ValueError("Every partition must contain at least one scored response")
    train = expected["train"]
    assert set(train["student_index"]).issubset(split["train"])
    assert set(train["qid"]).issubset(train_items)
    if protocol == "cold":
        assert set(train["qid"]).isdisjoint(expected["val"]["qid"])
        assert set(train["qid"]).isdisjoint(expected["test"]["qid"])
        assert set(expected["val"]["qid"]).isdisjoint(expected["test"]["qid"])

    metadata = {
        "schema_version": 1, "protocol": protocol,
        "source_sha256": {name: _digest(data_root / name) for name in ("used.txt", "used.json")},
        "semantic_dim": input_dim, "semantic_tables_frozen": True,
        "student_count": len(rows), "question_count": item_count,
        "raw_interactions": sum(len(q) for q, _ in rows),
        "split_seed": split_seed, "item_seed": item_seed if protocol == "cold" else None,
        "student_partition_counts": {key: len(ids) for key, ids in split.items()},
        "item_partition_counts": {key: len(ids) for key, ids in item_split.items()},
        "sequence_length": sequence_length, "train_step": train_step,
        "eval_step": eval_step if protocol == "warm" else None,
        "training_policy": "Each filtered training response is scored once per epoch; right padding is unscored.",
        "evaluation_policy": (
            "Overlapping windows score each raw response once; the first window has 0..L-1 past responses, "
            "later targets have L-step..L-1 preceding responses. No recurrent state crosses windows."
            if protocol == "warm" else
            "A cold query sees at most L-1 preceding train-item responses from the same held-out student. "
            "All validation/test item responses are excluded from every history. Only the single query is scored."
        ),
        "audits": {
            "student_partitions_disjoint": True, "target_coverage_exact_once": True,
            "teacher_uses_training_students_only": True,
            "teacher_uses_training_items_only": True,
            "no_previous_cold_response_in_history": True if protocol == "cold" else None,
            "cold_item_partitions_disjoint": True if protocol == "cold" else None,
        },
        "partitions": {},
    }
    for key, target in expected.items():
        metadata["partitions"][key] = {
            "windows": len(datasets[key]), "targets": int(len(target["labels"])),
            "positives": int(target["labels"].sum()),
            "unique_students_with_targets": int(len(np.unique(target["student_index"]))),
            "unique_items_with_targets": int(len(np.unique(target["qid"]))),
            "target_order_sha256": _identity_digest(target),
        }
        if protocol == "cold" and key != "train":
            counts = [min(query[1], sequence_length - 1) for query in datasets[key].queries]
            metadata["partitions"][key]["queries_without_history"] = counts.count(0)
            metadata["partitions"][key]["mean_prior_train_responses"] = float(np.mean(counts))
    return {
        **datasets, "rows": rows, "protocol_rows": protocol_rows,
        "split": split, "item_split": item_split, "expected": expected,
        "question_table": question_table, "concept_table": concept_table,
        "input_dim": input_dim, "train_item_ids": list(item_split["train"]),
        "train_student_ids": list(split["train"]),
        "train_student_index": train["student_index"],
        "train_question_ids": train["qid"], "train_labels": train["labels"],
        "metadata": metadata,
    }
