"""Audit and report the complete prespecified 18-run Transformer comparison.

Run from the repository root after the full suite completes::

    python tools/summarize_transformer.py --root runs/transformer-irt-20261013 \
        --output reports/transformer-irt

Until all planned summaries exist, --status prints completion counts only.
Incomplete suites cannot produce a metric report or open prediction archives.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("warm", "cold")
VARIANTS = ("original", "static", "aligned")
SEEDS = (42, 2026, 3407)
METRICS = ("auc", "acc", "nll", "brier", "ece15")
IDENTITY_KEYS = ("student_index", "interaction_index", "qid", "labels")
NAMES = {"original": "原始 Transformer/AKT", "static": "静态语义题目参数",
         "aligned": "静态参数 + 训练集难度对齐"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_hashes(value):
    return {name.replace("\\", "/"): digest for name, digest in value.items()}


def current_source_hashes():
    # Same manifest as transformer_experiments.run, without importing PyTorch.
    paths = (sorted((ROOT / "experiments").glob("*.py")) +
             sorted((ROOT / "transformer_experiments").glob("*.py")) +
             [ROOT / "model" / "my_model.py"])
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def protocol_sha256(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def identity_sha256(arrays):
    digest = hashlib.sha256()
    for key in IDENTITY_KEYS:
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(arrays[key], dtype="<i8").tobytes())
    return digest.hexdigest()


def checked_metrics(probabilities, labels):
    """Independently recompute all five archived test metrics, with strict input checks."""
    p, y = np.asarray(probabilities, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    if p.ndim != 1 or y.shape != p.shape or not len(p):
        raise ValueError("Invalid prediction/label shapes")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid probabilities or binary labels")
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        raise ValueError("AUC requires both response classes")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    ranks = (starts + ends + 1) / 2
    auc = (np.dot(ranks, np.add.reduceat(sorted_y, starts)) - positives * (positives + 1) / 2) / (positives * negatives)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    bins = np.minimum((p * 15).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(p[bins == b].mean() - y[bins == b].mean())
              for b in range(15) if np.any(bins == b))
    return {"auc": float(auc), "acc": float(np.mean((p >= .5) == y)),
            "nll": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            "brier": float(np.mean((p - y) ** 2)), "ece15": float(ece), "count": len(y)}


def assert_metrics(actual, saved, description):
    if actual["count"] != saved.get("count"):
        raise ValueError(f"{description}: target count differs")
    for name in METRICS:
        value = saved.get(name)
        if value is None or not math.isfinite(float(value)) or not math.isclose(
                actual[name], float(value), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{description}: recomputed {name} differs from summary")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) != len(SEEDS) or not np.isfinite(values).all():
        raise ValueError("Every reported statistic requires the three planned finite observations")
    return {"n": len(values), "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def planned_runs(root):
    root = Path(root)
    return {(protocol, variant, seed): root / f"{protocol}-{variant}-s{seed}"
            for protocol in PROTOCOLS for variant in VARIANTS for seed in SEEDS}


def completion_status(root):
    """Expose only existence/counts; do not deserialize test results."""
    planned = planned_runs(root)
    complete = sum(all((path / name).is_file() for name in
                       ("config.json", "summary.json", "best.pt", "test_predictions.npz"))
                   for path in planned.values())
    return {"planned_runs": len(planned), "completed_artifact_sets": complete,
            "pending_artifact_sets": len(planned) - complete,
            "all_planned_artifacts_present": complete == len(planned),
            "meaning": "Artifact presence only; metric and integrity audits require the complete suite."}


def aggregate(root, data_root):
    root, data_root = Path(root), Path(data_root)
    status = completion_status(root)
    if not status["all_planned_artifacts_present"]:
        raise ValueError(f"Incomplete planned suite: {status['completed_artifact_sets']}/18 artifact sets; no test metrics read")
    sources = current_source_hashes()
    data_hashes = {name: file_sha256(data_root / name) for name in ("used.txt", "used.json")}
    protocols = {protocol: read_json(root / f"{protocol}-protocol.json") for protocol in PROTOCOLS}
    records, common_recipe = {}, None
    variant_recipes = {}
    # Validate every completed summary and all configurations before opening any predictions.
    for key, directory in planned_runs(root).items():
        protocol, variant, seed = key
        config, summary = read_json(directory / "config.json"), read_json(directory / "summary.json")
        if config.get("smoke") is not False or not isinstance(summary.get("test"), dict):
            raise ValueError(f"Smoke or incomplete formal result: {directory}")
        if summary.get("config") != config or tuple(config.get(k) for k in ("protocol", "variant", "seed")) != key:
            raise ValueError(f"Configuration/identity mismatch: {directory}")
        if normalize_hashes(config.get("source_sha256", {})) != sources:
            raise ValueError(f"Current source hashes differ: {directory}")
        if config.get("data_sha256") != data_hashes or protocols[protocol].get("source_sha256") != data_hashes:
            raise ValueError(f"Current input hashes differ: {directory}")
        if config.get("protocol_sha256") != protocol_sha256(protocols[protocol]):
            raise ValueError(f"Protocol metadata hash differs: {directory}")
        if config.get("anchor_weight") != 0 or config.get("alignment_weight") != (.05 if variant == "aligned" else 0):
            raise ValueError(f"Prespecified auxiliary-loss recipe changed: {directory}")
        if config.get("backbone") != "original_TextAKT3PL_2blocks_8heads":
            raise ValueError(f"Unexpected backbone: {directory}")
        if (variant == "aligned") != isinstance(config.get("teacher_metadata"), dict):
            raise ValueError(f"Teacher configuration mismatch: {directory}")
        ignored = {"protocol", "variant", "seed", "protocol_sha256", "alignment_weight", "teacher_metadata"}
        recipe = {k: v for k, v in config.items() if k not in ignored}
        recipe["source_sha256"] = normalize_hashes(recipe["source_sha256"])
        if common_recipe is None:
            common_recipe = recipe
        elif recipe != common_recipe:
            raise ValueError(f"Common training settings changed across planned runs: {directory}")
        variant_recipe = {k: v for k, v in config.items() if k != "seed"}
        variant_recipe["source_sha256"] = normalize_hashes(variant_recipe["source_sha256"])
        previous = variant_recipes.setdefault((protocol, variant), variant_recipe)
        if previous != variant_recipe:
            raise ValueError(f"Variant/teacher settings changed across seeds: {directory}")
        if not (1 <= summary["best_epoch"] <= summary["epochs_run"] <= config["epochs"]):
            raise ValueError(f"Invalid selected epoch: {directory}")
        checkpoint_hash = file_sha256(directory / "best.pt")
        if checkpoint_hash != summary.get("checkpoint_sha256"):
            raise ValueError(f"Checkpoint SHA-256 differs: {directory}")
        records[key] = {"path": directory, "config": config, "summary": summary,
                        "checkpoint_sha256": checkpoint_hash}
    parameter_counts = {record["summary"]["trainable_parameters"] for record in records.values()}
    if len(parameter_counts) != 1:
        raise ValueError("The controlled Transformer variants must have identical parameter counts")
    audits, run_details = {}, []
    for protocol in PROTOCOLS:
        reference = None
        metadata = protocols[protocol]["partitions"]["test"]
        for variant in VARIANTS:
            for seed in SEEDS:
                record = records[(protocol, variant, seed)]
                directory, summary = record["path"], record["summary"]
                with np.load(directory / "test_predictions.npz", allow_pickle=False) as archive:
                    arrays = {name: np.array(archive[name], copy=True) for name in IDENTITY_KEYS}
                    probabilities = np.asarray(archive["probabilities"])
                    if any(value.ndim != 1 or value.shape != arrays["labels"].shape for value in arrays.values()):
                        raise ValueError(f"Invalid test identity shapes: {directory}")
                    for name, value in arrays.items():
                        if not np.isfinite(value).all() or not np.array_equal(value, value.astype(np.int64)):
                            raise ValueError(f"Noninteger test identity {name}: {directory}")
                    if reference is None:
                        reference = arrays
                    elif any(not np.array_equal(reference[name], arrays[name]) for name in IDENTITY_KEYS):
                        raise ValueError(f"Test identities/order changed across variants or seeds: {directory}")
                    identity_hash = identity_sha256(arrays)
                    if identity_hash != metadata["target_order_sha256"] or len(probabilities) != metadata["targets"]:
                        raise ValueError(f"Test identities disagree with hashed protocol: {directory}")
                    recomputed = checked_metrics(probabilities, arrays["labels"])
                assert_metrics(recomputed, summary["test"], str(directory))
                record["test"] = recomputed
                run_details.append({"protocol": protocol, "variant": variant, "seed": seed,
                                    "run_directory": directory.as_posix(), "best_epoch": summary["best_epoch"],
                                    "epochs_run": summary["epochs_run"], "seconds": summary["seconds"],
                                    "trainable_parameters": summary["trainable_parameters"],
                                    "validation": summary["validation"], "test": recomputed,
                                    "checkpoint_sha256": record["checkpoint_sha256"],
                                    "prediction_archive_sha256": file_sha256(directory / "test_predictions.npz"),
                                    "config_file_sha256": file_sha256(directory / "config.json"),
                                    "summary_file_sha256": file_sha256(directory / "summary.json"),
                                    "target_identity_sha256": identity_hash})
        audits[protocol] = {"passed": True, "completed_runs": 9, "protocol_sha256": protocol_sha256(protocols[protocol]),
                            "test_target_count": metadata["targets"], "target_identity_sha256": metadata["target_order_sha256"],
                            "all_seeds_and_variants_have_identical_targets": True,
                            "all_five_metrics_recomputed": True, "all_checkpoints_hash_verified": True}
    groups = [{"protocol": protocol, "variant": variant, "seeds": list(SEEDS),
               "test": {metric: stats([records[(protocol, variant, seed)]["test"][metric] for seed in SEEDS])
                        for metric in METRICS}}
              for protocol in PROTOCOLS for variant in VARIANTS]
    comparisons = []
    for protocol in PROTOCOLS:
        for before, after in (("original", "static"), ("static", "aligned"), ("original", "aligned")):
            pairs = [{"seed": seed, "baseline_auc": records[(protocol, before, seed)]["test"]["auc"],
                      "candidate_auc": records[(protocol, after, seed)]["test"]["auc"],
                      "delta": {metric: records[(protocol, after, seed)]["test"][metric] -
                                         records[(protocol, before, seed)]["test"][metric] for metric in METRICS}}
                     for seed in SEEDS]
            for pair in pairs:
                pair["delta_auc_percentage_points"] = pair["delta"]["auc"] * 100
            comparisons.append({"protocol": protocol, "contrast": f"{after} - {before}",
                                "baseline_variant": before, "candidate_variant": after, "paired_seeds": list(SEEDS),
                                "delta_auc_percentage_points": stats([pair["delta_auc_percentage_points"] for pair in pairs]),
                                "delta_metrics": {metric: stats([pair["delta"][metric] for pair in pairs]) for metric in METRICS},
                                "pairs": pairs})
    return {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "design": "original Transformer/AKT, static semantic item heads, and +0.05 train-only Rasch alignment",
            "completion_status": {**status, "all_planned_18_runs_audited": True},
            "common_training_recipe": common_recipe, "source_sha256": sources, "data_sha256": data_hashes,
            "comparability_audits": audits, "groups": groups, "paired_comparisons": comparisons, "runs": run_details,
            "limitations": [
                "三组均保留原始 Transformer/AKT、3PL 输出及题目参数参与的交互门控；参数量和初始化顺序相同。",
                "static 仅将三个题目参数头的输入改为位置编码和上下文注意力之前的语义投影；aligned 仅额外加入 0.05 倍训练集 Rasch 难度对齐，不增加锚定正则。",
                "AKT 的能力输出仍依赖当前题目的查询，且原始题目相关门控仍存在；本实验不能宣称完整心理测量解耦或全局单调性。",
                "原始 256 维题干、256 维知识点向量保持固定，不新增可训练题目 ID 嵌入；Rasch 教师只使用训练学生、训练题目的作答。",
                "每组先按验证集 AUC 选权重，再评估测试集；本汇总在全部 18 组完成后才读取测试预测。",
                "均值与样本标准差描述三个训练种子的变化，不是显著性检验或置信区间，也未集成不同种子的预测。",
                "cold 只使用一次固定题目划分；多训练种子不能替代多次题目划分验证。warm 与 cold 的目标与历史条件不同，不能直接比较二者数值。",
                "与保留的 80.6453% 集成结果的训练学生数、模型形式、历史窗口协议不同，不能直接解释为同协议涨跌。",
                "warm 测试学生划分曾用于此前项目开发；本轮不用于训练或选权重，但并非从未查看过的全新外部测试集。",
            ]}


def format_stat(value, scale=1):
    return f"{value['mean'] * scale:.4f} ± {value['sample_std'] * scale:.4f}"


def markdown(report):
    lines = ["# 原始 Transformer/AKT 的语义题目参数实验", "",
             f"生成时间（UTC）：{report['generated_at_utc']}。计划的 18 组正式实验全部完成并通过归档核验。", "",
             "三个模型均保留原始 Transformer/AKT。static 让题目难度、区分度、猜测率只由当前题目语义产生；aligned 在 static 上增加训练集难度对齐。", "",
             "各训练种子的算术均值 ± 样本标准差（ddof=1）；AUC、ACC 为百分比，NLL、Brier、ECE15 越低越好。", "",
             "| 协议 | 模型 | AUC（%） | ACC（%） | NLL | Brier | ECE15 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in report["groups"]:
        values = group["test"]
        cells = [format_stat(values[name], 100 if name in {"auc", "acc"} else 1) for name in METRICS]
        lines.append(f"| {group['protocol']} | {NAMES[group['variant']]} | " + " | ".join(cells) + " |")
    lines += ["", "## 同种子配对增量", "", "AUC 差值单位为百分点，正值表示候选模型更高。三个对照分别检验静态参数、额外对齐、两者合计的变化。", "",
              "| 协议 | 候选 − 对照 | 平均增量 ± 样本标准差 | 各种子增量 |", "|---|---|---:|---|"]
    for contrast in report["paired_comparisons"]:
        pairs = "; ".join(f"{pair['seed']}: {pair['delta_auc_percentage_points']:+.4f}" for pair in contrast["pairs"])
        lines.append(f"| {contrast['protocol']} | {contrast['contrast']} | {format_stat(contrast['delta_auc_percentage_points'])} | {pairs} |")
    lines += ["", "## 逐次实验", "", "| 协议 | 模型 | 种子 | 最佳轮次 / 实际轮次 | 验证 AUC（%） | 测试 AUC（%） | 耗时（秒） |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        lines.append(f"| {run['protocol']} | {run['variant']} | {run['seed']} | {run['best_epoch']} / {run['epochs_run']} | "
                     f"{run['validation']['auc'] * 100:.4f} | {run['test']['auc'] * 100:.4f} | {run['seconds']:.1f} |")
    lines += ["", "## 核验与解释边界", "",
              "已核对当前代码与原始数据哈希、协议元数据哈希、每组配置与权重哈希；同协议全部九组的标签、学生、题目、交互位置及顺序逐项一致。五项测试指标全部从归档预测重新计算，与训练汇总一致。完整哈希及五项指标的配对差值见同名 JSON。", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/transformer-irt-20261013")
    parser.add_argument("--data-root", default=str(ROOT / "dataset" / "NIPS2020"))
    parser.add_argument("--output", help="Report filename stem, required unless --status")
    parser.add_argument("--status", action="store_true", help="Completion counts only; no test metrics read")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(completion_status(args.root), ensure_ascii=False))
        return
    if not args.output:
        parser.error("--output is required unless --status")
    try:
        report = aggregate(args.root, args.data_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Report refused: {error}\n")
    output = Path(args.output)
    if output.suffix in {".json", ".md"}:
        output = output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = Path(str(output) + ".json"), Path(str(output) + ".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"completed_runs": 18, "all_audits_passed": True,
                      "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
