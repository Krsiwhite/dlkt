"""Posthoc difficulty diagnostics for the completed 18-run Transformer suite.

The complete suite must pass the existing source/data/checkpoint/target/metric
audit before any held-out Rasch reference is fitted. Static semantic parameters
are also checked against the selected checkpoint on CPU. No response predictions,
training, checkpoint selection or hyperparameter updates are performed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.data import prepare  # noqa: E402
from transformer_experiments.models import build_model  # noqa: E402
from tools import evaluate_item_difficulty as diagnostics  # noqa: E402
from tools import summarize_transformer as suite_audit  # noqa: E402

SUBSETS = ("all_observed_items", "reliable_items_n_ge_30")
CORRELATIONS = ("spearman", "pearson")


def paired_comparisons(records):
    """Compare aligned minus static for all three prespecified paired seeds."""
    indexed = {(r["protocol"], r["variant"], r["seed"]): r for r in records}
    expected = {(p, v, s) for p in suite_audit.PROTOCOLS
                for v in ("static", "aligned") for s in suite_audit.SEEDS}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Difficulty comparisons require all 12 unique planned static/aligned runs")
    comparisons = []
    for protocol in suite_audit.PROTOCOLS:
        for subset in SUBSETS:
            for metric in CORRELATIONS:
                pairs = []
                for seed in suite_audit.SEEDS:
                    baseline = indexed[(protocol, "static", seed)][subset]
                    candidate = indexed[(protocol, "aligned", seed)][subset]
                    if baseline["n_items"] != candidate["n_items"]:
                        raise ValueError("Paired correlations must use the same reference items")
                    before, after = baseline[metric], candidate[metric]
                    if any(x is not None and not np.isfinite(x) for x in (before, after)):
                        raise ValueError("Correlation must be finite or explicitly undefined")
                    pairs.append({"seed": seed, "static": before, "aligned": after,
                                  "delta": None if before is None or after is None else after - before})
                finite = [p["delta"] for p in pairs if p["delta"] is not None]
                comparisons.append({"protocol": protocol, "subset": subset, "metric": metric,
                                    "contrast": "aligned - static", "paired_seeds": list(suite_audit.SEEDS),
                                    "n_finite_pairs": len(finite),
                                    "mean_delta": float(np.mean(finite)) if finite else None,
                                    "sample_sd_delta": float(np.std(finite, ddof=1)) if len(finite) >= 2 else None,
                                    "pairs": pairs})
    return comparisons


def load_verified_parameters(directory, config, summary, bundle):
    """Bind every saved b/a/guess value to the selected model's static heads."""
    directory = Path(directory)
    checkpoint_path, parameter_path = directory / "best.pt", directory / "item_parameters.npz"
    if suite_audit.file_sha256(checkpoint_path) != summary["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed since suite audit: {directory}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"] != config or checkpoint["epoch"] != summary["best_epoch"]:
        raise ValueError(f"Checkpoint configuration/epoch mismatch: {directory}")
    if checkpoint["validation"] != summary["validation"]:
        raise ValueError(f"Checkpoint validation selection mismatch: {directory}")
    n_items = len(bundle["question_table"]) - 1
    saved = {}
    with np.load(parameter_path, allow_pickle=False) as archive:
        if set(archive.files) != {"b", "a", "guess"}:
            raise ValueError(f"Expected all three Transformer item parameters: {parameter_path}")
        for name in ("b", "a", "guess"):
            value = np.asarray(archive[name], dtype=np.float64)
            if value.shape != (n_items, 1) or not np.isfinite(value).all():
                raise ValueError(f"Invalid saved parameter shape or values: {name}")
            saved[name] = value.reshape(-1)
    if (np.any(np.abs(saved["b"]) > 3.000001) or np.any(saved["a"] < .099999)
            or np.any((saved["guess"] < 0) | (saved["guess"] > 1))):
        raise ValueError("Saved parameters violate the original Transformer head bounds")
    model = build_model(config["variant"], input_dim=bundle["input_dim"],
                        model_dim=config["model_dim"], seq_len=config["sequence_length"]).cpu().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    with torch.inference_mode():
        calculated = model.item_parameters(bundle["concept_table"][1:], bundle["question_table"][1:])
    differences = {}
    for name in ("b", "a", "guess"):
        actual = calculated[name].numpy().astype(np.float64).reshape(-1)
        # Saved parameters were evaluated on ROCm; verification is CPU only.
        if not np.allclose(saved[name], actual, rtol=1e-5, atol=1e-5):
            raise ValueError(f"Saved {name} does not match selected checkpoint's static head: {directory}")
        differences[name] = float(np.max(np.abs(saved[name] - actual)))
    return saved, {"verified_against_selected_checkpoint": True,
                   "verification_device": "cpu", "rtol": 1e-5, "atol": 1e-5,
                   "max_absolute_difference": differences,
                   "item_parameters_sha256": suite_audit.file_sha256(parameter_path)}


def evaluate(root, data_root, reference_steps=300):
    root, data_root = Path(root), Path(data_root)
    # aggregate() checks existence of every planned artifact before reading ANY
    # test metrics. It then verifies all 18 sources, inputs, checkpoints, target
    # identities and archived metrics. Do not move dataset loading above it.
    audited = suite_audit.aggregate(root, data_root)
    records, references = [], []
    for protocol in suite_audit.PROTOCOLS:
        template_dir = root / f"{protocol}-static-s{suite_audit.SEEDS[0]}"
        template = diagnostics.read_json(template_dir / "config.json")
        bundle = prepare(data_root, protocol, sequence_length=template["sequence_length"],
                         split_seed=template["split_seed"], item_seed=template["item_seed"],
                         eval_step=template["eval_step"])
        if (bundle["metadata"]["source_sha256"] != template["data_sha256"] or
                suite_audit.protocol_sha256(bundle["metadata"]) != template["protocol_sha256"]):
            raise ValueError("Prepared data/protocol differs from the completed training suite")
        # Confirm that real prepared target identities match those already
        # verified against every prediction archive by the suite audit.
        expected_hash = suite_audit.identity_sha256(bundle["expected"]["test"])
        if expected_hash != audited["comparability_audits"][protocol]["target_identity_sha256"]:
            raise ValueError("Prepared test identities differ from completed predictions")
        checked = []
        for variant in ("static", "aligned"):
            for seed in suite_audit.SEEDS:
                directory = root / f"{protocol}-{variant}-s{seed}"
                config = diagnostics.read_json(directory / "config.json")
                summary = diagnostics.read_json(directory / "summary.json")
                values, parameter_audit = load_verified_parameters(directory, config, summary, bundle)
                checked.append((variant, seed, directory, summary, values, parameter_audit))
        reference = diagnostics.build_reference(bundle, protocol, reference_steps)
        references.append(reference["metadata"])
        item_ids = reference["item_ids"]
        reliable_ids = item_ids[reference["counts"][item_ids] >= 30]
        for variant, seed, directory, summary, values, parameter_audit in checked:
            records.append({"protocol": protocol, "variant": variant, "seed": seed,
                            "run": directory.name, "checkpoint_sha256": summary["checkpoint_sha256"],
                            "parameter_archive_audit": parameter_audit,
                            "all_observed_items": diagnostics.correlations(values["b"][item_ids], reference["raw"][item_ids]),
                            "reliable_items_n_ge_30": diagnostics.correlations(values["b"][reliable_ids], reference["raw"][reliable_ids])})
        del bundle
    return {"schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Exploratory Transformer static-difficulty validity against a held-out Rasch proxy; not psychological ground truth.",
            "selection_policy": "All 18 planned runs and checkpoints fixed before this diagnostic; reference labels never update or select a model.",
            "new_response_prediction_forward_passes": 0,
            "static_parameter_verification_forward_passes": len(records),
            "correlation_weighting": "Equal item weighting; average-rank Spearman handles ties.",
            "between_seed_sd": "Sample SD (ddof=1), not a sampling confidence interval.",
            "completed_suite_audit": {"completion_status": audited["completion_status"],
                                      "comparability_audits": audited["comparability_audits"],
                                      "source_sha256": audited["source_sha256"], "data_sha256": audited["data_sha256"]},
            "diagnostic_source_sha256": {Path(p).resolve().relative_to(ROOT).as_posix(): suite_audit.file_sha256(p)
                                          for p in (__file__, diagnostics.__file__, suite_audit.__file__)},
            "references": references, "runs": records, "aggregates": diagnostics.aggregate(records),
            "paired_comparisons": paired_comparisons(records),
            "additional_limitations": [
                "No single invariant difficulty is defined for the contextual original variant; only static and aligned are compared here.",
                "The original Transformer ability remains current-question-conditioned, and item parameters still enter interaction gates.",
                "A fixed-slope static Rasch proxy differs from the model's 3PL response equation and dynamic student states.",
                "Cold uses one fixed item partition; three training seeds do not measure uncertainty across item partitions.",
                "Higher difficulty correlation alone does not establish improved response prediction or validity of discrimination/guess/ability."]}


def markdown(report):
    text = diagnostics.markdown_report(report)
    text = text.replace("# 题目难度独立有效性诊断", "# 原始 Transformer 上的题目难度诊断", 1)
    text = text.replace("没有重新预测、训练或选择模型。", "没有重新进行作答预测、训练或选择模型。CPU 静态参数头前向计算仅用于核对归档参数与选定权重一致。", 1)
    lines = [text, "## 同种子对齐增量", "",
             "以下为 aligned − static 的相关系数差值，不是 AUC 差值。正值表示与独立 Rasch 参照更一致。", "",
             "| 协议 | 题目子集 | 相关指标 | 平均差值 ± SD | 各种子差值 |",
             "|---|---|---|---:|---|"]
    def number(value):
        return "NA" if value is None else f"{value:+.4f}"
    for comparison in report["paired_comparisons"]:
        subset = "全部可观察题目" if comparison["subset"] == SUBSETS[0] else "每题 n≥30"
        pairs = "; ".join(f"{p['seed']}: {number(p['delta'])}" for p in comparison["pairs"])
        sd = comparison["sample_sd_delta"]
        sd_text = "NA" if sd is None else f"{sd:.4f}"
        lines.append(f"| {comparison['protocol']} | {subset} | {comparison['metric']} | "
                     f"{number(comparison['mean_delta'])} ± {sd_text} | {pairs} |")
    lines.extend(["", "原始 original 组的难度依赖位置及上下文，没有同题唯一的静态难度，本诊断不为其构造任意聚合值。"
                  "Transformer 的能力仍以当前题目为条件，题目参数仍参与门控；本诊断不能证明完整心理测量解耦。", ""])
    return "\n".join(lines)


def self_test():
    """Synthetic-only checks, including refusal before dataset/test access."""
    diagnostics.self_test()
    with tempfile.TemporaryDirectory() as temporary:
        try:
            evaluate(temporary, Path(temporary) / "nonexistent-data")
        except ValueError as error:
            assert "Incomplete planned suite" in str(error)
        else:
            raise AssertionError("Incomplete suite was not rejected")
        # Synthetic semantic tensors and a fresh toy checkpoint exercise the
        # archive-to-selected-weight link; no real dataset or saved run is used.
        directory = Path(temporary)
        config = {"variant": "static", "input_dim": 16, "model_dim": 8, "sequence_length": 7}
        model = build_model("static", input_dim=16, model_dim=8, seq_len=7).cpu().eval()
        bundle = {"input_dim": 16, "question_table": torch.randn(5, 16),
                  "concept_table": torch.randn(5, 16)}
        with torch.inference_mode():
            params = {name: tensor.numpy() for name, tensor in
                      model.item_parameters(bundle["concept_table"][1:], bundle["question_table"][1:]).items()}
        torch.save({"config": config, "epoch": 1, "validation": {"auc": .5},
                    "model": model.state_dict()}, directory / "best.pt")
        summary = {"checkpoint_sha256": suite_audit.file_sha256(directory / "best.pt"),
                   "best_epoch": 1, "validation": {"auc": .5}}
        np.savez(directory / "item_parameters.npz", **params)
        _, verified = load_verified_parameters(directory, config, summary, bundle)
        assert verified["verified_against_selected_checkpoint"]
        assert max(verified["max_absolute_difference"].values()) == 0
        params["b"] = params["b"] + .25
        np.savez(directory / "item_parameters.npz", **params)
        try:
            load_verified_parameters(directory, config, summary, bundle)
        except ValueError as error:
            assert "does not match selected checkpoint" in str(error)
        else:
            raise AssertionError("Tampered item-parameter archive was accepted")
    records = []
    for protocol in suite_audit.PROTOCOLS:
        for variant in ("static", "aligned"):
            for seed in suite_audit.SEEDS:
                value = .2 if variant == "static" else .3
                records.append({"protocol": protocol, "variant": variant, "seed": seed,
                                **{subset: {"n_items": 10, "spearman": value, "pearson": value}
                                   for subset in SUBSETS}})
    comparisons = paired_comparisons(records)
    assert len(comparisons) == 8
    assert all(abs(c["mean_delta"] - .1) < 1e-12 and c["n_finite_pairs"] == 3 for c in comparisons)
    for invalid in (records[:-1], records + [records[0]]):
        try:
            paired_comparisons(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Missing or duplicate seed was accepted")
    print("Transformer synthetic pairing and incomplete-suite guards passed; no real dataset or test result read.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/transformer-irt-20261013")
    parser.add_argument("--data-root", default="dataset/NIPS2020")
    parser.add_argument("--output", default="reports/transformer-difficulty.json")
    parser.add_argument("--reference-steps", type=int, default=300)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.self_test:
        self_test()
        return
    output = Path(args.output)
    if output.suffix != ".json":
        parser.error("--output must end in .json; a sibling .md report is also written")
    if output.exists() or output.with_suffix(".md").exists():
        parser.error("Output already exists; choose a new report path to preserve provenance")
    report = evaluate(args.root, args.data_root, args.reference_steps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(output.with_suffix('.md')),
                      "completed_suite_runs_audited": 18, "static_parameter_runs_diagnosed": len(report["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
