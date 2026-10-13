"""Summarize completed, comparable experiments without opening unfinished tests.

Example (run from the repository root)::

    python tools/summarize_experiments.py \
        --roots runs/round1-s42 runs/round1-repeat --output reports/round1

This writes reports/round1.json and reports/round1.md. Prediction archives are
read only for completed decoupled/aligned pairs. Reports contain aggregate
metrics, hashes, and run metadata; no individual student/item response records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("auc", "nll", "brier", "ece15")
IDENTITY_KEYS = ("labels", "student_index", "qid", "interaction_index")
COMMON_RECIPE_KEYS = (
    "split_seed", "item_seed", "epochs", "patience", "batch_size",
    "sequence_length", "eval_step", "lr", "weight_decay", "model_dim",
    "new_model_dropout", "gradient_clip", "label_smoothing", "selection", "test_policy",
)
VARIANT_ORDER = {"original": 0, "semantic_mlp": 1, "decoupled": 2, "aligned": 3}
VARIANT_NAMES = {"original": "原始模型", "semantic_mlp": "语义 MLP 对照",
                 "decoupled": "题目参数与学生状态分离", "aligned": "分离模型 + 难度对齐"}


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Descriptive statistics require nonempty finite observations")
    return {"n": int(len(array)), "mean": float(array.mean()),
            "sample_std": float(array.std(ddof=1)) if len(array) >= 2 else None,
            "min": float(array.min()), "max": float(array.max())}


def auc(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Invalid prediction/label shapes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonfinite/out-of-range probabilities or nonbinary labels")
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both response classes")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    ranks = (starts + ends + 1) / 2
    positive_counts = np.add.reduceat(sorted_y, starts)
    return float((np.dot(ranks, positive_counts) - positives * (positives + 1) / 2)
                 / (positives * negatives))


def discover(roots):
    records, warnings, ignored_smoke = [], [], 0
    seen_files, seen_keys = set(), {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            warnings.append(f"尚无目录，未读取：{root.as_posix()}")
            continue
        summaries = sorted(root.rglob("summary.json"))
        for path in summaries:
            resolved = path.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # The trainer creates summary.json last; a partial write is
                # not evidence that the training/evaluation completed.
                warnings.append(f"汇总文件尚未完整写入，未读取：{path.as_posix()}")
                continue
            config = summary.get("config", {})
            if config.get("smoke") is True:
                ignored_smoke += 1
                continue
            if config.get("smoke") is not False or not isinstance(summary.get("test"), dict):
                warnings.append(f"不满足完整正式实验标记，已跳过：{path.as_posix()}")
                continue
            key = (config["protocol"], config["variant"], int(config["seed"]))
            if key in seen_keys:
                raise ValueError(f"Duplicate protocol/variant/seed {key}: {seen_keys[key]} and {path}")
            seen_keys[key] = path
            for field in ("source_sha256", "data_sha256", "protocol_sha256"):
                if not config.get(field):
                    raise ValueError(f"Completed run lacks {field}: {path}")
            for metric in METRICS:
                value = summary["test"].get(metric)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"Completed run lacks finite test {metric}: {path}")
            if not (1 <= summary["best_epoch"] <= summary["epochs_run"] <= config["epochs"]):
                raise ValueError(f"Invalid epoch record: {path}")
            if summary["test"].get("count", 0) < 1:
                raise ValueError(f"Invalid test target count: {path}")
            records.append({"path": path, "summary": summary, "config": config, "key": key})
        if not summaries:
            warnings.append(f"目录中尚无已完成汇总，未读取进行中的实验：{root.as_posix()}")
    records.sort(key=lambda r: (r["key"][0], VARIANT_ORDER.get(r["key"][1], 99), r["key"][2]))
    return records, warnings, ignored_smoke


def verify_comparability(records):
    """Fail rather than silently average changing code/data/experimental rules."""
    by_protocol = defaultdict(list)
    for record in records:
        by_protocol[record["key"][0]].append(record)
    audits = {}
    for protocol, subset in sorted(by_protocol.items()):
        first = subset[0]
        first_config = first["config"]
        reference = {
            "source_sha256": first_config["source_sha256"],
            "data_sha256": first_config["data_sha256"],
            "protocol_sha256": first_config["protocol_sha256"],
            "common_training_recipe": {key: first_config[key] for key in COMMON_RECIPE_KEYS},
            "test_target_count": first["summary"]["test"]["count"],
        }
        for record in subset[1:]:
            config = record["config"]
            candidate = {
                "source_sha256": config["source_sha256"],
                "data_sha256": config["data_sha256"],
                "protocol_sha256": config["protocol_sha256"],
                "common_training_recipe": {key: config[key] for key in COMMON_RECIPE_KEYS},
                "test_target_count": record["summary"]["test"]["count"],
            }
            for key, expected in reference.items():
                if candidate[key] != expected:
                    raise ValueError(f"Incomparable {protocol} runs: {key} differs between "
                                     f"{first['path']} and {record['path']}")
        # Variant-specific settings and teacher targets must also stay fixed
        # across seeds, although they legitimately differ between variants.
        by_variant = defaultdict(list)
        for record in subset:
            by_variant[record["key"][1]].append(record)
        variant_configs = {}
        for variant, runs in by_variant.items():
            config = {k: v for k, v in runs[0]["config"].items() if k != "seed"}
            for record in runs[1:]:
                other = {k: v for k, v in record["config"].items() if k != "seed"}
                if other != config:
                    raise ValueError(f"Variant configuration/teacher changed across seeds: {protocol}/{variant}")
            variant_configs[variant] = canonical_sha256(config)
        audits[protocol] = {
            "passed": True, "compared_completed_runs": len(subset), **reference,
            "variant_config_excluding_seed_sha256": variant_configs,
        }
    return audits


def verify_pair(decoupled, aligned):
    left_path = decoupled["path"].parent / "test_predictions.npz"
    right_path = aligned["path"].parent / "test_predictions.npz"
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        identity = hashlib.sha256()
        for key in IDENTITY_KEYS:
            if key not in left.files or key not in right.files:
                raise ValueError(f"Paired prediction archive lacks {key}")
            if not np.array_equal(left[key], right[key]):
                raise ValueError(f"Paired test {key} differs: {left_path} vs {right_path}")
            if left[key].ndim != 1:
                raise ValueError(f"Paired identity array {key} is not one dimensional")
            identity.update(key.encode("ascii"))
            identity.update(np.asarray(left[key], dtype="<i8").tobytes())
        labels = left["labels"]
        if any(len(left[key]) != len(labels) for key in IDENTITY_KEYS):
            raise ValueError("Paired identity arrays have inconsistent lengths")
        if len(labels) != decoupled["summary"]["test"]["count"]:
            raise ValueError("Paired archive count differs from completed summary")
        before = auc(left["probabilities"], labels)
        after = auc(right["probabilities"], labels)
        for value, record in ((before, decoupled), (after, aligned)):
            if not math.isclose(value, record["summary"]["test"]["auc"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Archived prediction AUC disagrees with summary: {record['path']}")
        return {
            "seed": decoupled["key"][2], "targets": len(labels),
            "identity_arrays_exact_match": True,
            "target_identity_sha256": identity.hexdigest(),
            "decoupled_predictions_sha256": file_sha256(left_path),
            "aligned_predictions_sha256": file_sha256(right_path),
            "decoupled_auc": before, "aligned_auc": after,
            "delta_auc_percentage_points": (after - before) * 100,
        }


def aggregate(records, warnings, ignored_smoke, roots):
    audits = verify_comparability(records)
    grouped = defaultdict(list)
    lookup = {}
    for record in records:
        grouped[record["key"][:2]].append(record)
        lookup[record["key"]] = record
    groups = []
    for (protocol, variant), runs in sorted(grouped.items(), key=lambda x: (x[0][0], VARIANT_ORDER.get(x[0][1], 99))):
        groups.append({
            "protocol": protocol, "variant": variant,
            "seeds": [run["key"][2] for run in runs], "seed_count": len(runs),
            "test": {metric: stats([run["summary"]["test"][metric] for run in runs]) for metric in METRICS},
            "seconds": stats([run["summary"]["seconds"] for run in runs]),
        })
    comparisons = []
    for protocol in sorted(audits):
        available_before = {key[2] for key in lookup if key[:2] == (protocol, "decoupled")}
        available_after = {key[2] for key in lookup if key[:2] == (protocol, "aligned")}
        paired_seeds = sorted(available_before & available_after)
        unpaired_seeds = sorted(available_before ^ available_after)
        if unpaired_seeds:
            warnings.append(f"{protocol} 尚未形成完整 decoupled/aligned 配对的种子：{unpaired_seeds}")
        if not paired_seeds:
            continue
        pairs = [verify_pair(lookup[(protocol, "decoupled", seed)], lookup[(protocol, "aligned", seed)])
                 for seed in paired_seeds]
        comparisons.append({
            "protocol": protocol, "contrast": "aligned - decoupled",
            "unit": "AUC percentage points", "paired_seed_count": len(pairs),
            "paired_seeds": paired_seeds,
            "delta_auc_percentage_points": stats([pair["delta_auc_percentage_points"] for pair in pairs]),
            "pairs": pairs,
            "interpretation": "Descriptive variation across training seeds; no significance test, confidence interval, or prediction ensemble.",
        })
    run_details = []
    for record in records:
        summary, config = record["summary"], record["config"]
        run_details.append({
            "protocol": config["protocol"], "variant": config["variant"], "seed": config["seed"],
            "summary_file": record["path"].as_posix(),
            "best_epoch": summary["best_epoch"], "epochs_run": summary["epochs_run"],
            "seconds": summary["seconds"], "trainable_parameters": summary["trainable_parameters"],
            "validation": {metric: summary["validation"][metric] for metric in METRICS},
            "test": {**{metric: summary["test"][metric] for metric in METRICS}, "count": summary["test"]["count"]},
            "checkpoint_sha256": summary["checkpoint_sha256"],
        })
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_roots": [Path(root).as_posix() for root in roots],
        "completed_run_count": len(records), "ignored_smoke_run_count": ignored_smoke,
        "three_paired_seeds_available_for_both_protocols": (
            {item["protocol"] for item in comparisons if item["paired_seed_count"] >= 3} >= {"warm", "cold"}
        ),
        "comparability_audits": audits, "groups": groups,
        "paired_alignment_comparisons": comparisons, "runs": run_details,
        "warnings": warnings,
        "limitations": [
            "仅纳入已写入完整 summary.json、smoke=false 且具有测试结果的实验；不读取未完成实验的测试预测。",
            "均值与样本标准差描述训练随机种子的变化。三个种子仍然很少，不构成显著性结论；单一种子没有可估计的标准差。",
            "AUC 先对每个种子单独计算，再取算术平均；未融合不同种子的预测。配对差值按同一个种子逐一计算。",
            "原始模型和语义 MLP 若仅一个种子，只能作为初步对照，其结果不具备多种子稳健性证据。",
            "warm 与 cold 的目标题目、交互记录和历史条件不同，二者 AUC 不可直接作优劣比较。",
            "本实验与保留的 80.6453% 结果在训练学生数、模型/集成形式、窗口历史规则等方面不同，不可直接解释为相同协议下的涨跌。",
            "当前 cold 是一次固定题目划分；训练种子复验不能替代多次题目划分验证。",
        ],
    }


def format_stat(value, scale=1, digits=4):
    mean = value["mean"] * scale
    suffix = (f" ± {value['sample_std'] * scale:.{digits}f}"
              if value["sample_std"] is not None else "（单种子）")
    return f"{mean:.{digits}f}{suffix}"


def markdown(report):
    complete = report["three_paired_seeds_available_for_both_protocols"]
    lines = ["# 第一轮语义 IRT 实验汇总", "",
             f"生成时间（UTC）：{report['generated_at_utc']}。已完成正式实验 {report['completed_run_count']} 组。",
             "", "状态：" + ("warm、cold 均已有至少三个种子的分离/对齐配对。" if complete else
                                "阶段性汇总，部分计划实验或种子尚未完成；空缺不代表零效果。"), "",
             "## 测试集结果", "",
             "下表为各训练种子的算术均值 ± 样本标准差（ddof=1）。AUC 以百分比显示，标准差以百分点显示；NLL、Brier、ECE15 越低越好。", "",
             "| 协议 | 模型 | 种子数 | AUC（%） | NLL | Brier | ECE15 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        metric = group["test"]
        lines.append(f"| {group['protocol']} | {VARIANT_NAMES.get(group['variant'], group['variant'])} | "
                     f"{group['seed_count']} | {format_stat(metric['auc'], 100)} | {format_stat(metric['nll'])} | "
                     f"{format_stat(metric['brier'])} | {format_stat(metric['ece15'])} |")
    if not report["groups"]:
        lines.append("| — | 尚无已完成正式实验 | 0 | — | — | — | — |")
    lines += ["", "## 难度对齐的配对增量", "",
              "差值为同一种子的 aligned AUC − decoupled AUC，单位为百分点。归档预测中的标签、学生、题目和原交互位置均逐项核对一致，且重新计算 AUC 与汇总值一致。", "",
              "| 协议 | 配对种子 | 平均增量 ± 样本标准差（百分点） | 各种子增量（百分点） |",
              "|---|---|---:|---|"]
    for item in report["paired_alignment_comparisons"]:
        per_seed = "; ".join(f"{pair['seed']}: {pair['delta_auc_percentage_points']:+.4f}" for pair in item["pairs"])
        lines.append(f"| {item['protocol']} | {', '.join(map(str, item['paired_seeds']))} | "
                     f"{format_stat(item['delta_auc_percentage_points'])} | {per_seed} |")
    if not report["paired_alignment_comparisons"]:
        lines.append("| — | 暂无完整配对 | — | — |")
    lines += ["", "这里没有做显著性检验、置信区间或跨种子预测集成。三个种子只能提供有限的随机性证据。", "",
              "## 各实验记录", "",
              "最佳轮次只根据验证集 AUC 选择；测试集在选定权重之后评估。耗时是训练器记录的本次模型训练、验证和最终测试时间，不含套件启动和教师拟合时间。", "",
              "| 协议 | 模型 | 种子 | 最佳轮次 / 实际轮次 | 验证 AUC（%） | 测试 AUC（%） | 耗时（秒） |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        lines.append(f"| {run['protocol']} | {run['variant']} | {run['seed']} | {run['best_epoch']} / {run['epochs_run']} | "
                     f"{run['validation']['auc'] * 100:.4f} | {run['test']['auc'] * 100:.4f} | {run['seconds']:.1f} |")
    lines += ["", "## 可比性核验", ""]
    for protocol, audit in report["comparability_audits"].items():
        lines.append(f"- {protocol}：{audit['compared_completed_runs']} 组已完成实验的代码哈希、数据哈希、协议哈希、共同训练设置和测试目标数一致；"
                     f"测试目标 {audit['test_target_count']:,} 条。各模型的种子间设置与教师目标也保持一致。")
    if not report["comparability_audits"]:
        lines.append("尚无可核验的已完成正式实验。")
    lines += ["", "完整哈希和逐种子指标保存在同名 JSON 文件中；报告未保存任何单条学生作答记录。", "",
              "## 结论边界", ""]
    lines += ["- " + item for item in report["limitations"]]
    if report["warnings"]:
        lines += ["", "## 当前未纳入项", ""]
        lines += ["- " + warning for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output filename stem, e.g. reports/round1")
    args = parser.parse_args()
    records, warnings, ignored_smoke = discover(args.roots)
    report = aggregate(records, warnings, ignored_smoke, args.roots)
    output = Path(args.output)
    if output.suffix in {".json", ".md"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(str(output) + ".json")
    markdown_path = Path(str(output) + ".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"completed_runs": len(records), "ignored_smoke_runs": ignored_smoke,
                      "paired_protocols": [item["protocol"] for item in report["paired_alignment_comparisons"]],
                      "json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
