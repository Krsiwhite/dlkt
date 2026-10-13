"""Run prespecified ablations; checkpoint selection never reads test predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.data import prepare
from transformer_experiments.models import build_model
from experiments.teacher import fit_teacher, alignment_loss


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def metrics(p, y):
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(p) != len(y) or not len(p) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction vector")
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_p)) + 1]
    ends = np.r_[starts[1:], len(p)]
    ranks = (starts + ends + 1) / 2
    pos_by_group = np.add.reduceat(sorted_y, starts)
    positives, negatives = y.sum(), len(y) - y.sum()
    auc = float((np.dot(ranks, pos_by_group) - positives * (positives + 1) / 2) / (positives * negatives)) if positives and negatives else None
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    buckets = np.minimum((p * 15).astype(int), 14)
    ece = sum(np.mean(buckets == b) * abs(p[buckets == b].mean() - y[buckets == b].mean()) for b in range(15) if np.any(buckets == b))
    return {"auc": auc, "acc": float(np.mean((p >= .5) == y)),
            "nll": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()),
            "brier": float(np.mean((p - y) ** 2)), "ece15": float(ece), "count": len(y)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Do not promise bitwise reproducibility across ROCm training runs.
    torch.use_deterministic_algorithms(False)


def loader(dataset, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator, pin_memory=False)


def batch_to_device(batch, device):
    if isinstance(batch, dict):
        qids, responses, mask = batch["qids"], batch["responses"], batch["score_mask"]
    else:
        qids, responses, mask = batch
    # Right padding cannot affect preceding causal predictions.
    length = int((qids != 0).sum(1).max())
    return (qids[:, :length].to(device), responses[:, :length].to(device), mask[:, :length].to(device))


@torch.inference_mode()
def evaluate(model, data_loader, question_table, concept_table, expected, device, max_batches=None):
    model.eval()
    probabilities, labels = [], []
    for number, batch in enumerate(data_loader):
        if max_batches is not None and number >= max_batches:
            break
        qids, responses, mask = batch_to_device(batch, device)
        p = model(concept_table[qids], question_table[qids], responses.unsqueeze(-1)).squeeze(-1)
        if not torch.isfinite(p[mask]).all():
            raise ValueError("Non-finite evaluation predictions")
        probabilities.append(p[mask].cpu().numpy())
        labels.append(responses[mask].cpu().numpy())
    p, y = np.concatenate(probabilities), np.concatenate(labels)
    if max_batches is None:
        target = np.asarray(expected["labels"])
        if not np.array_equal(y, target):
            raise AssertionError("Evaluation target ordering/count mismatch")
    return metrics(p, y), p


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "experiments").glob("*.py")) + sorted((root / "transformer_experiments").glob("*.py")) + [root / "model" / "my_model.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_one(args, bundle, protocol, variant, seed, teacher, device):
    seed_all(seed)
    directory = Path(args.output) / f"{protocol}-{variant}-s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    config = {"protocol": protocol, "variant": variant, "seed": seed, "split_seed": 42,
              "item_seed": 20261013, "epochs": args.epochs, "patience": args.patience,
              "batch_size": args.batch_size, "sequence_length": args.sequence_length,
              "eval_step": args.eval_step, "lr": args.lr, "weight_decay": 1e-4,
              "alignment_weight": args.alignment_weight if variant == "aligned" else 0,
              "anchor_weight": 0, "backbone": "original_TextAKT3PL_2blocks_8heads",
              "model_dim": 128, "akt_dropout": 0.0, "semantic_context_dropout": .3, "gradient_clip": 1,
              "label_smoothing": 0, "selection": "maximum validation AUC, then earlier epoch",
              "test_policy": "once after validation selection", "smoke": args.smoke,
              "data_sha256": bundle["metadata"]["source_sha256"],
              "protocol_sha256": hashlib.sha256(json.dumps(bundle["metadata"], sort_keys=True).encode()).hexdigest(),
              "teacher_metadata": teacher["metadata"] if variant == "aligned" else None,
              "source_sha256": source_hashes()}
    if (directory / "summary.json").exists():
        old = json.loads((directory / "summary.json").read_text())
        if old["config"] != config:
            raise ValueError(f"Existing run configuration mismatch: {directory}")
        print(json.dumps({"event": "already_complete", "run": directory.name}), flush=True)
        return old
    if (directory / "epochs.jsonl").exists():
        raise RuntimeError(f"Incomplete run exists: {directory}. Use a new output directory; do not mix histories.")
    write_json(directory / "config.json", config)
    model = build_model(variant, model_dim=128, seq_len=args.sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    qtable, ctable = bundle["question_table"].to(device), bundle["concept_table"].to(device)
    assert not qtable.requires_grad and not ctable.requires_grad
    train_loader = loader(bundle["train"], args.batch_size, True, seed)
    val_loader = loader(bundle["val"], args.batch_size)
    best_auc, best_epoch, best_val = -math.inf, 0, None
    began = time.time()
    print(json.dumps({"event": "start", "run": directory.name, "parameters": sum(p.numel() for p in model.parameters()),
                      "train_batches": len(train_loader), "validation_batches": len(val_loader)}), flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_bce, total_aux, targets, batches = 0., 0., 0, 0
        for number, batch in enumerate(train_loader):
            if args.smoke and number >= 2:
                break
            qids, responses, mask = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(ctable[qids], qtable[qids], responses.unsqueeze(-1)).squeeze(-1)
            main_loss = F.binary_cross_entropy(pred[mask].clamp(1e-7, 1 - 1e-7), responses[mask])
            auxiliary = torch.zeros((), device=device)
            if variant == "aligned":
                auxiliary = auxiliary + args.alignment_weight * alignment_loss(model, ctable, qtable, teacher)
            loss = main_loss + auxiliary
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("Non-finite gradient norm")
            optimizer.step()
            n = int(mask.sum())
            total_bce += main_loss.item() * n
            total_aux += auxiliary.item()
            targets += n
            batches += 1
        scheduler.step()
        val, _ = evaluate(model, val_loader, qtable, ctable, bundle["expected"]["val"], device,
                          max_batches=2 if args.smoke else None)
        if val["auc"] is None:
            raise ValueError("Validation set has only one label class")
        if val["auc"] > best_auc:
            best_auc, best_epoch, best_val = val["auc"], epoch, val
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config, "validation": val}, directory / "best.pt")
        record = {"epoch": epoch, "train_bce": total_bce / targets, "train_auxiliary": total_aux / batches,
                  "targets": targets, "validation": val, "seconds": time.time() - start, "best_epoch": best_epoch,
                  "lr_next": scheduler.get_last_lr()[0]}
        with (directory / "epochs.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "config": config}, directory / "latest.pt")
        print(json.dumps({"event": "epoch", "run": directory.name, **record}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if args.smoke:
        test = None
    else:
        test_loader = loader(bundle["test"], args.batch_size)
        test, probabilities = evaluate(model, test_loader, qtable, ctable, bundle["expected"]["test"], device)
        np.savez_compressed(directory / "test_predictions.npz", probabilities=probabilities,
                            **bundle["expected"]["test"])
    item_diagnostics = None
    if variant in ("static", "aligned"):
        model.eval()
        with torch.no_grad():
            params = model.item_parameters(ctable[1:], qtable[1:])
            item_diagnostics = {key: {"mean": value.mean().item(), "std": value.std().item(),
                                       "min": value.min().item(), "max": value.max().item()}
                                for key, value in params.items()}
            np.savez_compressed(directory / "item_parameters.npz", **{key: value.cpu().numpy() for key, value in params.items()})
    summary = {"config": config, "best_epoch": best_epoch, "epochs_run": epoch, "validation": best_val,
               "test": test, "seconds": time.time() - began, "item_parameter_summary": item_diagnostics,
               "trainable_parameters": sum(p.numel() for p in model.parameters()),
               "checkpoint_sha256": hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()}
    write_json(directory / "summary.json", summary)
    print(json.dumps({"event": "complete", "run": directory.name, "best_epoch": best_epoch, "test": test,
                      "seconds": summary["seconds"]}), flush=True)
    del model, optimizer, scheduler, qtable, ctable
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/NIPS2020")
    parser.add_argument("--output", default="runs/transformer-irt-20261013")
    parser.add_argument("--protocols", nargs="+", default=["warm", "cold"], choices=["warm", "cold"])
    parser.add_argument("--variants", nargs="+", default=["original", "static", "aligned"],
                        choices=["original", "static", "aligned"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=201)
    parser.add_argument("--eval-step", type=int, default=100)
    parser.add_argument("--lr", type=float, default=.0005)
    parser.add_argument("--alignment-weight", type=float, default=.05)
    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(6)
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "device": str(device),
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "deterministic_algorithms": False, "arguments": vars(args), "source_sha256": source_hashes()}
    if (output / "environment.json").exists():
        old_environment = json.loads((output / "environment.json").read_text())
        if old_environment != environment:
            raise ValueError("Existing suite environment or arguments differ; use a new output directory")
    else:
        write_json(output / "environment.json", environment)
    summaries = []
    for protocol in args.protocols:
        bundle = prepare(args.data_root, protocol, sequence_length=args.sequence_length, eval_step=args.eval_step)
        write_json(output / f"{protocol}-protocol.json", bundle["metadata"])
        print(json.dumps({"event": "data_prepared", "protocol": protocol, "metadata": bundle["metadata"]}), flush=True)
        teacher = None
        if "aligned" in args.variants:
            teacher_path = output / f"{protocol}-teacher.pt"
            if teacher_path.exists():
                teacher = torch.load(teacher_path, weights_only=True, map_location=device)
                fingerprint = hashlib.sha256()
                for array in (bundle["train_student_index"].astype(np.int64),
                              bundle["train_question_ids"].astype(np.int64), bundle["train_labels"].astype(np.uint8)):
                    fingerprint.update(array.tobytes())
                if (teacher["metadata"]["steps_requested"] != args.teacher_steps or
                    teacher["metadata"]["training_records_sha256"] != fingerprint.hexdigest()):
                    raise ValueError("Cached teacher does not match this training dataset or configuration")
            else:
                teacher = fit_teacher(bundle["train_student_index"], bundle["train_question_ids"], bundle["train_labels"],
                                      len(bundle["question_table"]) - 1, device=str(device), steps=args.teacher_steps)
                torch.save(teacher, teacher_path)
            write_json(output / f"{protocol}-teacher.json", teacher["metadata"])
        for seed in args.seeds:
            for variant in args.variants:
                summaries.append(run_one(args, bundle, protocol, variant, seed, teacher, device))
                write_json(output / "all_summaries.json", summaries)
    print(json.dumps({"event": "suite_complete", "runs": len(summaries), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
