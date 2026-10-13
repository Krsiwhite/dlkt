"""Post-hoc difficulty validity against an independent held-out Rasch proxy.

Invoke only after all prespecified runs have completed. No predictions, weight
updates, model selection, or test-based hyperparameter decisions are performed.
Saved semantic item parameters are compared with a reference fitted from held-out
test students. The reference is an exploratory model-based proxy, not true
psychological difficulty. Only aggregate metrics and fingerprints are exported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from experiments.data import prepare  # noqa: E402
from experiments.teacher import fit_teacher  # noqa: E402


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def average_ranks(values):
    """One-based average ranks, including exact ties; no SciPy dependency."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ranking requires a finite one-dimensional vector")
    if not len(values):
        return values.copy()
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return result


def pearson(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Correlation vectors must be one-dimensional and equally sized")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Correlation vectors must contain only finite values")
    if len(first) < 2:
        return None
    centered_first, centered_second = first - first.mean(), second - second.mean()
    denominator = np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    if denominator <= 0:
        return None
    return float(np.clip(np.dot(centered_first, centered_second) / denominator, -1, 1))


def correlations(predicted, reference):
    return {
        "n_items": int(len(predicted)),
        "pearson": pearson(predicted, reference),
        "spearman": pearson(average_ranks(predicted), average_ranks(reference)),
    }


def verify_complete_suites(roots):
    """Fail before reading dataset records if any planned run is unfinished."""
    selected, seen = [], set()
    for root in map(Path, roots):
        environment = read_json(root / "environment.json")
        arguments = environment["arguments"]
        if arguments.get("smoke", False):
            raise ValueError(f"Smoke suite is not eligible: {root}")
        expected = {(p, v, int(s)) for p in arguments["protocols"]
                    for v in arguments["variants"] for s in arguments["seeds"]}
        summaries = read_json(root / "all_summaries.json")
        indexed = {(s["config"]["protocol"], s["config"]["variant"], int(s["config"]["seed"])): s
                   for s in summaries}
        if len(indexed) != len(summaries) or set(indexed) != expected:
            raise ValueError(f"Suite has missing/duplicate/unplanned run summaries: {root}")
        for identity in sorted(expected):
            protocol, variant, seed = identity
            directory = root / f"{protocol}-{variant}-s{seed}"
            summary = read_json(directory / "summary.json")
            config = read_json(directory / "config.json")
            if summary != indexed[identity] or summary["config"] != config:
                raise ValueError(f"Inconsistent saved run/config: {directory}")
            if config.get("smoke") or not summary.get("test"):
                raise ValueError(f"Run has no completed final test evaluation: {directory}")
            if config.get("selection") != "maximum validation AUC, then earlier epoch":
                raise ValueError(f"Unexpected checkpoint selection rule: {directory}")
            if config.get("test_policy") != "once after validation selection":
                raise ValueError(f"Unexpected test policy: {directory}")
            if config["source_sha256"] != environment["source_sha256"]:
                raise ValueError(f"Run/suite source mismatch: {directory}")
            for relative, expected_hash in config["source_sha256"].items():
                source = (PROJECT_ROOT / relative).resolve()
                if not source.is_relative_to(PROJECT_ROOT):
                    raise ValueError("Invalid source path in saved configuration")
                if file_hash(source) != expected_hash:
                    raise ValueError(f"Frozen experiment source has changed: {relative}")
            if file_hash(directory / "best.pt") != summary["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {directory}")
            if variant in ("decoupled", "aligned"):
                if identity in seen:
                    raise ValueError(f"Duplicate protocol/variant/seed across suites: {identity}")
                seen.add(identity)
                if not (directory / "item_parameters.npz").is_file():
                    raise ValueError(f"Missing saved item parameters: {directory}")
                selected.append({"directory": directory, "suite": root.name,
                                 "summary": summary, "config": config})
    if not selected:
        raise ValueError("No completed decoupled/aligned runs found")
    return selected


def protocol_key(config):
    return (config["protocol"], config["split_seed"], config["item_seed"],
            config["sequence_length"], config["eval_step"], config["protocol_sha256"],
            tuple(sorted(config["data_sha256"].items())))


def build_reference(bundle, protocol, steps):
    """Fit evaluation-only reference on held-out students; never reuse it to train."""
    test_students = bundle["split"]["test"]
    if set(test_students) & (set(bundle["split"]["train"]) | set(bundle["split"]["val"])):
        raise ValueError("Held-out reference students overlap training or validation")
    n_items = len(bundle["question_table"]) - 1
    if protocol == "cold":
        train_items = set(bundle["item_split"]["train"])
        requested = np.asarray(bundle["item_split"]["test"], dtype=np.int64)
        allowed = train_items | set(requested)
        if (train_items & set(requested)) or (allowed & set(bundle["item_split"]["val"])):
            raise ValueError("Cold item partitions overlap")
        fitting_policy = "Held-out test students; train items plus test-cold items; validation-cold items excluded."
        scoring_policy = "Only test-cold items; these had zero response supervision in model training."
    else:
        requested = np.arange(n_items, dtype=np.int64)
        allowed = set(requested)
        fitting_policy = "Held-out test students; all items."
        scoring_policy = "All items observed among held-out test students."
    students, questions, labels = [], [], []
    for student in test_students:
        qids, outcomes = bundle["rows"][student]
        for question, label in zip(qids, outcomes):
            if question in allowed:
                students.append(student)
                questions.append(question)
                labels.append(label)
    teacher = fit_teacher(np.asarray(students, dtype=np.int64),
                          np.asarray(questions, dtype=np.int64),
                          np.asarray(labels, dtype=np.uint8), n_items,
                          device="cpu", steps=steps)
    counts = teacher["counts"].numpy()
    raw = teacher["raw_difficulty"].double().numpy()
    observed = requested[counts[requested] > 0]
    if not len(observed) or not np.isfinite(raw[observed]).all():
        raise ValueError("No finite independent difficulty reference on requested items")
    fit_metadata = dict(teacher["metadata"])
    # Generic fit_teacher names refer to its fit input; relabel their role here.
    fit_metadata.pop("teacher", None)
    fit_metadata["estimator"] = "regularized_rasch_1pl"
    fit_metadata["ability_constraint"] = "mean_reference_student_ability_zero"
    fit_metadata["reference_records_sha256"] = fit_metadata.pop("training_records_sha256")
    fit_metadata["reference_fit_nll"] = fit_metadata.pop("training_nll")
    metadata = {
        "protocol": protocol,
        "role": "exploratory_posthoc_heldout_reference_only",
        "fitting_policy": fitting_policy,
        "correlation_item_policy": scoring_policy,
        "n_requested_items": len(requested),
        "n_observed_requested_items": len(observed),
        "n_unobserved_requested_items": len(requested) - len(observed),
        "n_reliable_requested_items_n_ge_30": int(np.sum(counts[observed] >= 30)),
        "input_data_sha256": bundle["metadata"]["source_sha256"],
        "fit": fit_metadata,
        "reference_targets_exported": False,
        "reference_used_for_training_or_model_selection": False,
    }
    return {"raw": raw, "counts": counts, "item_ids": observed,
            "n_items": n_items, "metadata": metadata}


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["protocol"], record["variant"])].append(record)
    result = []
    for (protocol, variant), members in sorted(groups.items()):
        entry = {"protocol": protocol, "variant": variant,
                 "seeds": sorted(m["seed"] for m in members), "n_seeds": len(members)}
        for subset in ("all_observed_items", "reliable_items_n_ge_30"):
            item_counts = {m[subset]["n_items"] for m in members}
            if len(item_counts) != 1:
                raise ValueError("Cannot aggregate seeds evaluated on different item sets")
            stats = {"n_items": next(iter(item_counts))}
            for metric in ("spearman", "pearson"):
                values = [m[subset][metric] for m in members if m[subset][metric] is not None]
                stats[metric] = {"mean": float(np.mean(values)) if values else None,
                                 "sample_sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
                                 "n_finite_seeds": len(values)}
            entry[subset] = stats
        result.append(entry)
    return result


def markdown_report(report):
    def number(value):
        return "NA" if value is None else f"{value:.4f}"

    def mean_sd(stats):
        return number(stats["mean"]) + " ± " + number(stats["sample_sd"])

    lines = ["# 题目难度独立有效性诊断", "",
             "本报告在所有预定实验及权重选择完成后生成，仅比较已保存的题目参数；没有重新预测、训练或选择模型。",
             "参照难度来自独立测试学生的作答，采用固定斜率、带先验的 Rasch 拟合。它是探索性的行为代理指标，并非真实心理测量参数。",
             "冷启动协议拟合参照时只纳入训练题目与测试冷题，排除验证冷题；相关性仅计算测试冷题。测试参照从未作为训练、调参或权重选择输入。",
             "相关性每道题等权；Spearman 对并列值使用平均秩。平均值和样本标准差跨训练种子计算，不能解释为数据抽样置信区间。", "",
             "## 所有可观察目标题目", "",
             "| 协议 | 模型 | 种子数 | 题目数 | Spearman 均值 ± SD | Pearson 均值 ± SD |",
             "|---|---|---:|---:|---:|---:|"]
    for group in report["aggregates"]:
        stats = group["all_observed_items"]
        lines.append(f"| {group['protocol']} | {group['variant']} | {group['n_seeds']} | {stats['n_items']} | "
                     f"{mean_sd(stats['spearman'])} | {mean_sd(stats['pearson'])} |")
    lines.extend(["", "## 每题至少 30 次参照作答的敏感性检查", "",
                  "此子集结果只用于敏感性检查，不替代上表全部可观察题目的结果。", "",
                  "| 协议 | 模型 | 种子数 | 题目数 | Spearman 均值 ± SD | Pearson 均值 ± SD |",
                  "|---|---|---:|---:|---:|---:|"])
    for group in report["aggregates"]:
        stats = group["reliable_items_n_ge_30"]
        lines.append(f"| {group['protocol']} | {group['variant']} | {group['n_seeds']} | {stats['n_items']} | "
                     f"{mean_sd(stats['spearman'])} | {mean_sd(stats['pearson'])} |")
    lines.extend(["", "## 各种子结果", "",
                  "| 协议 | 模型 | 种子 | 全部题目 Spearman | 全部题目 Pearson | n≥30 Spearman | n≥30 Pearson |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for record in report["runs"]:
        all_items, reliable = record["all_observed_items"], record["reliable_items_n_ge_30"]
        lines.append(f"| {record['protocol']} | {record['variant']} | {record['seed']} | "
                     f"{number(all_items['spearman'])} | {number(all_items['pearson'])} | "
                     f"{number(reliable['spearman'])} | {number(reliable['pearson'])} |")
    lines.extend(["", "## 参照拟合与边界", ""])
    for reference in report["references"]:
        fit = reference["fit"]
        lines.append(f"- {reference['protocol']}：{fit['n_students']} 名测试学生，{fit['n_observations']} 条参照作答；"
                     f"目标题目 {reference['n_requested_items']} 道，可观察 {reference['n_observed_requested_items']} 道，"
                     f"未观察 {reference['n_unobserved_requested_items']} 道；拟合 {fit['steps_run']} 步，"
                     f"收敛标记 {fit['converged']}。")
    lines.extend(["", "Rasch 假设静态、单维能力，真实知识追踪存在学习和时间变化。较高相关性仅说明语义难度与此独立行为参照更一致，"
                  "不能单独证明心理测量有效性、因果解释或预测能力提升。这里不检验区分度或学生能力参数的有效性。",
                  "NA 表示相关性未定义或只有一个种子而无法估计样本标准差。公开结果仅包含汇总、配置校验与指纹，不包含学生记录或参照题目参数。", ""])
    return "\n".join(lines)


def self_test():
    """Synthetic-only checks; does not inspect suites or read the real dataset."""
    np.testing.assert_allclose(average_ranks([1, 1, 3, 2]), [1.5, 1.5, 4, 3])
    assert abs(pearson([1, 2, 3], [3, 2, 1]) + 1) < 1e-12
    assert correlations([1, 1, 3, 2], [4, 4, 6, 5])["spearman"] > 0.999999
    assert pearson([1, 1], [2, 3]) is None
    assert correlations([], [])["spearman"] is None
    students, questions, labels = [], [], []
    for student, ability in enumerate([-1, 0, 1]):
        for question, difficulty in enumerate([-1.5, 0, 1.5]):
            correct = round(100 / (1 + np.exp(-(ability - difficulty))))
            students.extend([student * 7] * 100)
            questions.extend([question] * 100)
            labels.extend([1] * correct + [0] * (100 - correct))
    reference = fit_teacher(students, questions, labels, 4, device="cpu")
    assert correlations([-1.5, 0, 1.5], reference["raw_difficulty"][:3].numpy())["spearman"] > 0.999999
    assert np.isnan(float(reference["raw_difficulty"][3]))
    print("Synthetic-only diagnostics checks passed; no run or dataset records were read.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", help="Completed run suite directories, e.g. first and repeat suites")
    parser.add_argument("--data-root", default="dataset/NIPS2020")
    parser.add_argument("--output", default="reports/difficulty-validity.json")
    parser.add_argument("--reference-steps", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        self_test()
        return
    if not args.roots:
        parser.error("--roots is required unless --self-test is used")
    selected = verify_complete_suites(args.roots)
    references, records, protocol_configs = {}, [], {}
    for run in selected:
        config, protocol = run["config"], run["config"]["protocol"]
        key = protocol_key(config)
        if protocol in protocol_configs and protocol_configs[protocol] != key:
            raise ValueError(f"Incompatible dataset/protocol configurations across seeds: {protocol}")
        protocol_configs[protocol] = key
        if protocol not in references:
            bundle = prepare(args.data_root, protocol, sequence_length=config["sequence_length"],
                             split_seed=config["split_seed"], item_seed=config["item_seed"],
                             eval_step=config["eval_step"])
            if bundle["metadata"]["source_sha256"] != config["data_sha256"]:
                raise ValueError("Input data differs from completed model training data")
            actual_protocol_hash = hashlib.sha256(json.dumps(bundle["metadata"], sort_keys=True).encode()).hexdigest()
            if actual_protocol_hash != config["protocol_sha256"]:
                raise ValueError("Prepared evaluation protocol differs from saved training configuration")
            references[protocol] = build_reference(bundle, protocol, args.reference_steps)
            del bundle
        reference = references[protocol]
        path = run["directory"] / "item_parameters.npz"
        with np.load(path, allow_pickle=False) as archive:
            if not {"b", "a"}.issubset(archive.files):
                raise ValueError(f"Missing item parameter arrays: {path}")
            values = {}
            for name in ("b", "a"):
                value = np.asarray(archive[name], dtype=np.float64)
                if value.shape not in ((reference["n_items"],), (reference["n_items"], 1)):
                    raise ValueError(f"Item parameter shape does not match dataset: {path}, {name}")
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite saved item parameters: {path}, {name}")
                values[name] = value.reshape(-1)
            if not (values["a"] > 0).all():
                raise ValueError(f"Saved discrimination is not positive: {path}")
        item_ids = reference["item_ids"]
        reliable_ids = item_ids[reference["counts"][item_ids] >= 30]
        records.append({"suite": run["suite"], "run": run["directory"].name,
                        "protocol": protocol, "variant": config["variant"], "seed": config["seed"],
                        "checkpoint_sha256": run["summary"]["checkpoint_sha256"],
                        "item_parameters_sha256": file_hash(path),
                        "source_sha256": config["source_sha256"],
                        "all_observed_items": correlations(values["b"][item_ids], reference["raw"][item_ids]),
                        "reliable_items_n_ge_30": correlations(values["b"][reliable_ids], reference["raw"][reliable_ids])})
    records.sort(key=lambda r: (r["protocol"], r["variant"], r["seed"]))
    report = {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
              "diagnostic_source_sha256": file_hash(__file__),
              "purpose": "Exploratory posthoc difficulty validity; independent held-out Rasch proxy, not psychological ground truth.",
              "selection_policy": "All runs complete before diagnostic; reference labels never select or update any model.",
              "new_prediction_forward_passes": 0,
              "correlation_weighting": "Equal item weighting; tie-aware average-rank Spearman.",
              "between_seed_sd": "Sample SD (ddof=1), not a confidence interval over sampled data.",
              "references": [references[p]["metadata"] for p in sorted(references)],
              "runs": records, "aggregates": aggregate(records)}
    output = Path(args.output)
    if output.suffix.lower() != ".json":
        raise ValueError("--output must be a .json path; a sibling .md report is also created")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(output.with_suffix('.md')),
                      "completed_runs": len(records), "protocols": sorted(references)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
