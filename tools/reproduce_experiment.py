"""Verify saved-checkpoint inference without retraining or changing run outputs.

Example (from the repository root):
    python tools/reproduce_experiment.py \
        --run-dir runs/round1-s42/warm-semantic_mlp-s42 \
        --data-root dataset/NIPS2020 --device cuda

The default report is RUN_DIR/reproduction.json. Existing files are never
overwritten; use --output with a fresh filename for a further verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.data import prepare
from experiments.models import build_model
from experiments.run import evaluate, loader, metrics, seed_all, source_hashes


def _read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_hashes(hashes: dict) -> dict:
    return {name.replace("\\", "/"): value for name, value in hashes.items()}


def _assert_equal(actual, expected, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description} mismatch; refusing to reproduce a different experiment")


def reproduce(run_dir: Path | str, data_root: Path | str,
              device: str = "cpu", output: Path | str | None = None) -> dict:
    """Audit inputs and evaluate an already completed checkpoint exactly once.

    No training, validation selection, or teacher fitting occurs. The sole
    write is a new JSON report opened in exclusive-create mode. A difference
    between CPU/GPU predictions is reported, not hidden behind a tolerance.
    """
    began = time.perf_counter()
    run_dir, data_root = Path(run_dir).resolve(), Path(data_root).resolve()
    output = (run_dir / "reproduction.json" if output is None else Path(output).resolve())
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}; choose a fresh --output path")
    needed = ("config.json", "summary.json", "best.pt", "test_predictions.npz")
    for name in needed:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Completed experiment artifact is missing: {run_dir / name}")
    config = _read_json(run_dir / "config.json")
    summary = _read_json(run_dir / "summary.json")
    if config.get("smoke") is not False or summary.get("test") is None:
        raise ValueError("Smoke or unfinished runs have no valid completed test result")
    _assert_equal(summary.get("config"), config, "Summary configuration")
    current_sources = _normalized_hashes(source_hashes())
    saved_sources = _normalized_hashes(config.get("source_sha256", {}))
    _assert_equal(current_sources, saved_sources, "Training source hashes")
    checkpoint_hash = _sha256(run_dir / "best.pt")
    _assert_equal(checkpoint_hash, summary.get("checkpoint_sha256"), "Checkpoint SHA-256")
    for name in ("used.txt", "used.json"):
        actual_hash = _sha256(data_root / name)
        _assert_equal(actual_hash, config.get("data_sha256", {}).get(name), f"Input {name} SHA-256")

    # These checkpoints contain tensors and plain configuration values only;
    # the restricted loader does not need arbitrary pickle object execution.
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    _assert_equal(checkpoint.get("config"), config, "Checkpoint configuration")
    _assert_equal(checkpoint.get("epoch"), summary.get("best_epoch"), "Selected epoch")
    _assert_equal(checkpoint.get("validation"), summary.get("validation"), "Selected validation result")

    bundle = prepare(data_root, config["protocol"],
                     sequence_length=config["sequence_length"],
                     split_seed=config["split_seed"], item_seed=config["item_seed"],
                     eval_step=config["eval_step"])
    protocol_hash = hashlib.sha256(
        json.dumps(bundle["metadata"], sort_keys=True).encode()).hexdigest()
    _assert_equal(protocol_hash, config.get("protocol_sha256"), "Reconstructed protocol SHA-256")
    expected = bundle["expected"]["test"]
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as archive:
        reference = np.array(archive["probabilities"], copy=True)
        for key in ("labels", "qid", "student_index", "interaction_index"):
            if key not in archive or not np.array_equal(archive[key], expected[key]):
                raise ValueError(f"Stored test target identity mismatch: {key}")
    if reference.ndim != 1 or reference.shape != expected["labels"].shape:
        raise ValueError("Stored test prediction shape mismatch")
    if not np.isfinite(reference).all() or not ((reference >= 0) & (reference <= 1)).all():
        raise ValueError("Stored test predictions must be finite probabilities")
    reference_metrics = metrics(reference, expected["labels"])
    _assert_equal(reference_metrics, summary["test"], "Stored prediction metrics")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm was requested but is unavailable in this environment")
    torch.set_num_threads(6)
    seed_all(config["seed"])
    model = build_model(config["variant"], input_dim=bundle["input_dim"],
                        model_dim=config["model_dim"], seq_len=config["sequence_length"],
                        dropout=config["new_model_dropout"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(target_device)
    reproduced_metrics, reproduced = evaluate(
        model, loader(bundle["test"], config["batch_size"]),
        bundle["question_table"].to(target_device), bundle["concept_table"].to(target_device),
        expected, target_device)
    if reproduced.shape != reference.shape:
        raise AssertionError("Reproduced prediction shape differs despite matching target identities")
    absolute_error = np.abs(reproduced.astype(np.float64) - reference.astype(np.float64))
    metric_differences = {
        key: (None if reproduced_metrics[key] is None or reference_metrics[key] is None
              else reproduced_metrics[key] - reference_metrics[key])
        for key in ("auc", "acc", "nll", "brier", "ece15")
    }
    report = {
        "schema_version": 1,
        "verification": "saved_checkpoint_test_inference_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "data_root": str(data_root),
        "protocol": config["protocol"], "variant": config["variant"],
        "seed": config["seed"], "best_epoch": summary["best_epoch"],
        "batch_size": config["batch_size"], "sequence_length": config["sequence_length"],
        "checkpoint_sha256": checkpoint_hash,
        "test_predictions_file_sha256": _sha256(run_dir / "test_predictions.npz"),
        "data_sha256": config["data_sha256"], "source_sha256": saved_sources,
        "protocol_sha256": protocol_hash,
        "audits": {
            "source_hashes_match": True, "data_hashes_match": True,
            "protocol_hash_matches": True, "checkpoint_config_matches": True,
            "checkpoint_hash_matches": True, "test_target_identities_match": True,
            "stored_predictions_match_summary_metrics": True,
            "original_artifacts_modified": False,
        },
        "predictions": {
            "count": int(len(reproduced)),
            "exact_equal": bool(np.array_equal(reproduced, reference)),
            "bitwise_equal": bool(reproduced.dtype == reference.dtype and
                                  reproduced.tobytes() == reference.tobytes()),
            "different_count": int(np.count_nonzero(reproduced != reference)),
            "max_absolute_difference": float(absolute_error.max()),
            "mean_absolute_difference": float(absolute_error.mean()),
            "reference_dtype": str(reference.dtype), "reproduced_dtype": str(reproduced.dtype),
        },
        "reference_metrics": reference_metrics,
        "reproduced_metrics": reproduced_metrics,
        "metric_differences": metric_differences,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(target_device),
            "gpu": (torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else None),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "seconds": time.perf_counter() - began,
        "scope": "Checks saved-checkpoint inference. Does not repeat optimization or establish bitwise training reproducibility.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset" / "NIPS2020")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        help="Fresh JSON report path; defaults to RUN_DIR/reproduction.json")
    args = parser.parse_args()
    try:
        report = reproduce(args.run_dir, args.data_root, args.device, args.output)
    except (OSError, ValueError, RuntimeError, KeyError, AssertionError) as error:
        parser.exit(1, f"Reproduction refused: {error}\n")
    destination = (args.run_dir / "reproduction.json" if args.output is None else args.output).resolve()
    print(json.dumps({"report": str(destination), "predictions": report["predictions"],
                      "reference_auc": report["reference_metrics"]["auc"],
                      "reproduced_auc": report["reproduced_metrics"]["auc"],
                      "scope": report["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
