"""Train-only Rasch supervision for a shared semantic difficulty head.

The teacher has temporary student/item parameters fitted from training responses.
Only fixed item targets are retained; these parameters are never model inputs or
deployment lookup tables. Rasch estimates are noisy model-based proxies, not
ground-truth psychological difficulty. Standardization retains relative difficulty
while fixing the scale used by the semantic alignment objective.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F


def _integer_vector(values: Any, name: str, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(values, device=device)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if value.is_floating_point():
        if not bool(torch.isfinite(value).all()) or not bool((value == value.round()).all()):
            raise ValueError(f"{name} must contain finite integer values")
    return value.to(torch.long)


@torch.no_grad()
def fit_teacher(
    student_ids: Any,
    question_ids: Any,
    labels: Any,
    n_items: int,
    device: str | torch.device = "cpu",
    steps: int = 300,
    *,
    prior_strength: float = 1.0,
    reliability_prior: float = 20.0,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Fit a fixed-slope, regularized 1PL teacher using *only* supplied records.

    Callers must pass training students and training questions only. Raw question
    IDs are zero-based; student IDs can be non-contiguous. The API deliberately
    has no validation/test inputs. Alternating Newton steps optimize Bernoulli
    likelihood with L2 priors, subject to mean training ability equal to zero.
    The fixed logistic slope fixes the Rasch scale; target standardization then
    sets the separate semantic head's alignment scale.

    Returned tensors are indexed by raw question ID and detached. Unobserved
    items have NaN difficulty, count zero, and weight zero. ``item_ids`` contains
    only observed training items. Reliability is n / (n + reliability_prior).
    """
    if n_items <= 0 or steps <= 0:
        raise ValueError("n_items and steps must be positive")
    if prior_strength <= 0 or reliability_prior < 0 or tolerance <= 0:
        raise ValueError("invalid regularization or convergence setting")
    target_device = torch.device(device)
    original_students = _integer_vector(student_ids, "student_ids", target_device)
    questions = _integer_vector(question_ids, "question_ids", target_device)
    dtype = torch.float64 if target_device.type == "cpu" else torch.float32
    outcomes = torch.as_tensor(labels, device=target_device, dtype=dtype)
    if outcomes.ndim != 1 or not (len(original_students) == len(questions) == len(outcomes)):
        raise ValueError("student_ids, question_ids, and labels must be equal-length vectors")
    if len(outcomes) == 0:
        raise ValueError("teacher requires at least one training observation")
    if bool((original_students < 0).any()):
        raise ValueError("student_ids must be nonnegative")
    if bool(((questions < 0) | (questions >= n_items)).any()):
        raise ValueError("question_ids must be in [0, n_items)")
    if not bool(((outcomes == 0) | (outcomes == 1)).all()):
        raise ValueError("labels must be binary and finite")

    active_students, student_inverse = torch.unique(original_students, sorted=True, return_inverse=True)
    n_students = len(active_students)
    counts = torch.bincount(questions, minlength=n_items)
    observed = counts > 0
    observed_ids = torch.nonzero(observed, as_tuple=False).flatten()
    correct = torch.bincount(questions, weights=outcomes, minlength=n_items)
    smoothed_rate = (correct + 0.5) / (counts.to(dtype) + 1.0)
    difficulty = -torch.logit(smoothed_rate)
    difficulty[~observed] = 0.0
    ability = torch.zeros(n_students, device=target_device, dtype=dtype)
    converged = False
    last_change = float("inf")

    for iteration in range(steps):
        probability = torch.sigmoid(ability[student_inverse] - difficulty[questions])
        information = probability * (1.0 - probability)
        ability_gradient = torch.bincount(
            student_inverse, weights=outcomes - probability, minlength=n_students
        ) - prior_strength * ability
        ability_hessian = torch.bincount(
            student_inverse, weights=information, minlength=n_students
        ) + prior_strength
        # Equality-constrained diagonal Newton step keeps mean ability at zero.
        multiplier = (ability_gradient / ability_hessian).sum() / (1.0 / ability_hessian).sum()
        ability_delta = ((ability_gradient - multiplier) / ability_hessian).clamp(-1.0, 1.0)
        ability_delta -= ability_delta.mean()
        ability += ability_delta

        probability = torch.sigmoid(ability[student_inverse] - difficulty[questions])
        information = probability * (1.0 - probability)
        difficulty_gradient = torch.bincount(
            questions, weights=probability - outcomes, minlength=n_items
        ) - prior_strength * difficulty
        difficulty_hessian = torch.bincount(
            questions, weights=information, minlength=n_items
        ) + prior_strength
        difficulty_delta = (difficulty_gradient / difficulty_hessian).clamp(-1.0, 1.0)
        difficulty_delta[~observed] = 0.0
        difficulty += difficulty_delta
        last_change = max(float(ability_delta.abs().max()), float(difficulty_delta.abs().max()))
        if iteration >= 4 and last_change < tolerance:
            converged = True
            break

    mean = difficulty[observed].mean()
    raw_std = difficulty[observed].std(unbiased=False)
    scale = raw_std.clamp_min(1e-6)
    targets = torch.full_like(difficulty, float("nan"))
    targets[observed] = (difficulty[observed] - mean) / scale
    raw_difficulty = difficulty.clone()
    raw_difficulty[~observed] = float("nan")
    weights = counts.to(dtype) / (counts.to(dtype) + reliability_prior).clamp_min(1.0)
    weights[~observed] = 0.0
    final_logits = ability[student_inverse] - difficulty[questions]
    training_nll = float(F.binary_cross_entropy_with_logits(final_logits, outcomes))
    fingerprint = hashlib.sha256()
    for vector in (original_students, questions, outcomes.to(torch.uint8)):
        fingerprint.update(vector.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        "teacher": "train_only_regularized_rasch_1pl",
        "logistic_slope": 1.0,
        "ability_constraint": "mean_training_ability_zero",
        "n_students": n_students,
        "n_items": n_items,
        "n_observed_items": len(observed_ids),
        "n_observations": len(outcomes),
        "prior_strength": prior_strength,
        "reliability_prior": reliability_prior,
        "steps_requested": steps,
        "steps_run": iteration + 1,
        "converged": converged,
        "last_parameter_change": last_change,
        "training_nll": training_nll,
        "raw_difficulty_mean": float(mean),
        "raw_difficulty_std": float(raw_std),
        "target_standardization": "unweighted_mean_zero_population_std_one_on_observed_items",
        "training_records_sha256": fingerprint.hexdigest(),
        "device": str(target_device),
        "interpretation": "Model-based behavior supervision; not ground-truth psychological parameters.",
    }
    return {
        "difficulty": targets.float().detach(),
        "raw_difficulty": raw_difficulty.float().detach(),
        "counts": counts.detach(),
        "weights": weights.float().detach(),
        "item_ids": observed_ids.detach(),
        "metadata": metadata,
    }


def alignment_loss(
    model: torch.nn.Module,
    concept_table: torch.Tensor,
    question_table: torch.Tensor,
    teacher: dict[str, Any],
    *,
    table_offset: int = 1,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Reliability-weighted Huber loss on observed training item difficulties.

    Semantic tables normally include padding at row zero (``table_offset=1``).
    Neither target responses nor teacher parameters are forwarded to the model.
    The model's shared item head must return a mapping containing ``b``.
    """
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    ids = torch.as_tensor(teacher["item_ids"], device=concept_table.device, dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("alignment requires at least one observed training item")
    table_ids = ids + table_offset
    if table_offset < 0 or bool((table_ids >= len(concept_table)).any()) or bool((table_ids >= len(question_table)).any()):
        raise ValueError("teacher item IDs do not match semantic lookup tables")
    prediction = model.item_parameters(concept_table[table_ids], question_table[table_ids])["b"].reshape(-1)
    if prediction.shape != ids.shape:
        raise ValueError("item_parameters must return one difficulty per observed item")
    targets = torch.as_tensor(teacher["difficulty"], device=prediction.device, dtype=prediction.dtype)[ids]
    weights = torch.as_tensor(teacher["weights"], device=prediction.device, dtype=prediction.dtype)[ids]
    if not bool(torch.isfinite(targets).all()) or not bool((weights > 0).all()):
        raise ValueError("observed teacher items must have finite targets and positive weights")
    losses = F.smooth_l1_loss(prediction, targets.detach(), reduction="none", beta=huber_delta)
    return (weights.detach() * losses).sum() / weights.sum().clamp_min(1e-12)
