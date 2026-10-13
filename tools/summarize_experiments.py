"""Summarize completed, comparable experiments without opening unfinished tests.

Example (run from the repository root)::

    python tools/summarize_experiments.py \
        --roots runs/round1-s42 runs/round1-repeat runs/round1-mlp-repeat \
        --output reports/round1

This writes reports/round1.json and reports/round1.md. Prediction archives are
read only for completed semantic_mlp/decoupled and decoupled/aligned pairs. Reports contain aggregate
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


def verify_pair(baseline, candidate):
    if baseline["key"][0] != candidate["key"][0] or baseline["key"][2] != candidate["key"][2]:
        raise ValueError("A prediction pair must use the same protocol and training seed")
    baseline_variant, candidate_variant = baseline["key"][1], candidate["key"][1]
    left_path = baseline["path"].parent / "test_predictions.npz"
    right_path = candidate["path"].parent / "test_predictions.npz"
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
        if len(labels) != baseline["summary"]["test"]["count"] or len(labels) != candidate["summary"]["test"]["count"]:
            raise ValueError("Paired archive count differs from completed summary")
        before = auc(left["probabilities"], labels)
        after = auc(right["probabilities"], labels)
        for value, record in ((before, baseline), (after, candidate)):
            if not math.isclose(value, record["summary"]["test"]["auc"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Archived prediction AUC disagrees with summary: {record['path']}")
        return {
            "seed": baseline["key"][2], "targets": len(labels),
            "baseline_variant": baseline_variant, "candidate_variant": candidate_variant,
            "identity_arrays_exact_match": True,
            "target_identity_sha256": identity.hexdigest(),
            baseline_variant + "_predictions_sha256": file_sha256(left_path),
            candidate_variant + "_predictions_sha256": file_sha256(right_path),
            baseline_variant + "_auc": before, candidate_variant + "_auc": after,
            "baseline_auc": before, "candidate_auc": after,
            "delta_auc_percentage_points": (after - before) * 100,
        }


def paired_contrast(lookup, protocols, baseline_variant, candidate_variant, warnings):
    comparisons = []
    for protocol in sorted(protocols):
        available_before = {key[2] for key in lookup if key[:2] == (protocol, baseline_variant)}
        available_after = {key[2] for key in lookup if key[:2] == (protocol, candidate_variant)}
        paired_seeds = sorted(available_before & available_after)
        unpaired_seeds = sorted(available_before ^ available_after)
        if unpaired_seeds:
            warnings.append(f"{protocol} 尚未形成完整 {baseline_variant}/{candidate_variant} 配对的种子：{unpaired_seeds}")
        if not paired_seeds:
            continue
        pairs = [verify_pair(lookup[(protocol, baseline_variant, seed)], lookup[(protocol, candidate_variant, seed)])
                 for seed in paired_seeds]
        comparisons.append({
            "protocol": protocol, "contrast": f"{candidate_variant} - {baseline_variant}",
            "baseline_variant": baseline_variant, "candidate_variant": candidate_variant,
            "unit": "AUC percentage points", "paired_seed_count": len(pairs),
            "paired_seeds": paired_seeds,
            "delta_auc_percentage_points": stats([pair["delta_auc_percentage_points"] for pair in pairs]),
            "pairs": pairs,
            "interpretation": "Descriptive variation across training seeds; no significance test, confidence interval, or prediction ensemble.",
        })
    return comparisons


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
    direction1 = paired_contrast(lookup, audits, "semantic_mlp", "decoupled", warnings)
    comparisons = paired_contrast(lookup, audits, "decoupled", "aligned", warnings)
    planned_seeds = {42, 2026, 3407}
    direction1_complete = {item["protocol"] for item in direction1
                           if planned_seeds.issubset(item["paired_seeds"])} >= {"warm", "cold"}
    direction2_complete = {item["protocol"] for item in comparisons
                           if planned_seeds.issubset(item["paired_seeds"])} >= {"warm", "cold"}
    original_complete = {key[0] for key in lookup if key[1] == "original"} >= {"warm", "cold"}
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
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_roots": [Path(root).as_posix() for root in roots],
        "completed_run_count": len(records), "ignored_smoke_run_count": ignored_smoke,
        "three_paired_seeds_available_for_both_protocols": (
            {item["protocol"] for item in comparisons if item["paired_seed_count"] >= 3} >= {"warm", "cold"}
        ),
        "completion_status": {
            "planned_training_seeds": sorted(planned_seeds),
            "planned_total_runs": 20,
            "direction1_all_planned_pairs_complete": direction1_complete,
            "direction2_all_planned_pairs_complete": direction2_complete,
            "original_reference_available_for_both_protocols": original_complete,
            "planned_20_run_design_complete": direction1_complete and direction2_complete and original_complete,
            "legacy_three_paired_seeds_field_covers_direction2_only": True,
        },
        "comparability_audits": audits, "groups": groups,
        "direction1_paired": direction1,
        "paired_alignment_comparisons": comparisons, "runs": run_details,
        "warnings": warnings,
        "limitations": [
            "仅纳入已写入完整 summary.json、smoke=false 且具有测试结果的实验；不读取未完成实验的测试预测。",
            "均值与样本标准差描述训练随机种子的变化。三个种子仍然很少，不构成显著性结论；单一种子没有可估计的标准差。",
            "AUC 先对每个种子单独计算，再取算术平均；未融合不同种子的预测。配对差值按同一个种子逐一计算。",
            "原始模型为单种子参考；任何尚未完成多种子的模型仅提供初步对照，不具备多种子稳健性证据。",
            "语义 MLP 的 2026、3407 种子复验是在观察到首个种子结果后追加的对照复验，沿用既定训练设置；属于事后补充验证，未进行超参数搜索。",
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
    complete = report["completion_status"]["planned_20_run_design_complete"]
    lines = ["# 第一轮语义 IRT 实验汇总", "",
             f"生成时间（UTC）：{report['generated_at_utc']}。已完成正式实验 {report['completed_run_count']} 组。",
             "", "状态：" + ("20 组计划实验已完成：warm、cold 的两个方向均包含 42、2026、3407 三种子配对，并各有单种子原始模型参考。" if complete else
                                "阶段性汇总，部分计划实验或种子尚未完成；空缺不代表零效果。完整计划为两个协议各 1 组原始模型及 3 个模型 × 3 个种子，共 20 组。"), "",
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
    for heading, description, items in (
        ("方向一：参数分离的配对增量",
         "差值为同一种子的 decoupled AUC − semantic_mlp AUC，即参数分离模型相对语义 MLP 对照的变化。",
         report["direction1_paired"]),
        ("方向二：难度对齐的配对增量",
         "差值为同一种子的 aligned AUC − decoupled AUC，即增加难度对齐监督后的变化。",
         report["paired_alignment_comparisons"]),
    ):
        lines += ["", "## " + heading, "", description + "单位为百分点；正值表示该方向提高 AUC。"
                  "归档预测中的标签、学生、题目和原交互位置均逐项核对一致，且重新计算 AUC 与汇总值一致。", "",
                  "| 协议 | 配对种子 | 平均增量 ± 样本标准差（百分点） | 各种子增量（百分点） |",
                  "|---|---|---:|---|"]
        for item in items:
            per_seed = "; ".join(f"{pair['seed']}: {pair['delta_auc_percentage_points']:+.4f}" for pair in item["pairs"])
            lines.append(f"| {item['protocol']} | {', '.join(map(str, item['paired_seeds']))} | "
                         f"{format_stat(item['delta_auc_percentage_points'])} | {per_seed} |")
        if not items:
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
                      "direction1_paired_protocols": [item["protocol"] for item in report["direction1_paired"]],
                      "paired_protocols": [item["protocol"] for item in report["paired_alignment_comparisons"]],
                      "planned_20_run_design_complete": report["completion_status"]["planned_20_run_design_complete"],
                      "json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
