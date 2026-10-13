"""Semantic-only controlled models for the two proposed research directions.

Question and concept identifiers never enter these modules. The runner indexes
immutable vector tables and supplies the resulting floating-point tensors.
``aligned`` is intentionally the exact same architecture as ``decoupled``;
only its train-only difficulty supervision differs in the runner.
"""

from __future__ import annotations

import importlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SemanticHistoryModel(nn.Module):
    """Shared GRU control and semantic 2PL with strictly history-only ability.

    Item parameters depend on the current fixed semantic vectors alone. The
    GRU state at prediction t contains interactions 0,...,t-1; the current
    question and current response cannot enter theta_t. States reset at the
    start of each supplied window. Right padding is safe when its predictions
    are excluded by the runner; left padding is not supported.

    The projections have no dropout or batch normalization, so item parameters
    stay identical across histories, positions and minibatches in training too.
    Dropout is restricted to the interaction/history and response-head paths.
    """

    def __init__(self, input_dim: int = 256, model_dim: int = 128,
                 dropout: float = 0.1, prediction_head: str = "2pl") -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("input_dim and model_dim must be positive")
        if prediction_head not in {"2pl", "mlp"}:
            raise ValueError(f"Unknown prediction head: {prediction_head}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.prediction_head = prediction_head

        def projection() -> nn.Sequential:
            return nn.Sequential(nn.Linear(input_dim, 2 * model_dim),
                                 nn.GELU(), nn.Linear(2 * model_dim, model_dim))

        self.concept_projection = projection()
        self.question_projection = projection()
        self.semantic_norm = nn.LayerNorm(model_dim)
        # Both controls encode a response with the same shared interaction map.
        # Correct/incorrect slots are shared globally, not per-item tables.
        self.interaction_projection = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim), nn.GELU(),
            nn.LayerNorm(model_dim), nn.Dropout(dropout))
        self.history = nn.GRU(model_dim, model_dim, num_layers=1, batch_first=True)

        hidden = max(8, model_dim // 2)
        if prediction_head == "2pl":
            self.difficulty_head = nn.Sequential(
                nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            self.discrimination_head = nn.Sequential(
                nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
            self.ability_head = nn.Sequential(
                nn.Linear(model_dim, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1))
            # Begin near a unit-discrimination, zero-difficulty model.
            nn.init.normal_(self.difficulty_head[-1].weight, std=0.02)
            nn.init.zeros_(self.difficulty_head[-1].bias)
            nn.init.normal_(self.discrimination_head[-1].weight, std=0.02)
            nn.init.constant_(self.discrimination_head[-1].bias,
                              math.log(math.expm1(0.9)))
        else:
            self.response_head = nn.Sequential(
                nn.Linear(2 * model_dim, model_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(model_dim, 1))

    def encode_items(self, concept_emb: Tensor, question_emb: Tensor) -> Tensor:
        if concept_emb.shape != question_emb.shape:
            raise ValueError("Concept and question vectors must have equal shapes")
        if concept_emb.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}")
        # Explicitly prohibit training the supplied raw vector tables. Shared
        # semantic projections still receive ordinary parameter gradients.
        return self.semantic_norm(
            self.concept_projection(concept_emb.detach())
            + self.question_projection(question_emb.detach()))

    def _history_before(self, semantic: Tensor, responses: Tensor) -> Tensor:
        if semantic.ndim != 3:
            raise ValueError("Sequence vectors must have shape [batch, length, dim]")
        if responses.ndim == 2:
            responses = responses.unsqueeze(-1)
        if responses.shape != semantic.shape[:2] + (1,):
            raise ValueError("Responses must have shape [batch, length, 1]")
        if semantic.shape[1] < 1:
            raise ValueError("An empty sequence is not supported")
        response = responses.detach().to(dtype=semantic.dtype)
        interaction = torch.cat((semantic * response,
                                 semantic * (1.0 - response)), dim=-1)
        interaction = self.interaction_projection(interaction)
        history_after, _ = self.history(interaction)
        # Shift STATES, rather than inserting a dummy interaction: the prior
        # for the first response is exactly the shared zero hidden state.
        return torch.cat((history_after.new_zeros(history_after.shape[0], 1,
                                                  self.model_dim),
                          history_after[:, :-1]), dim=1)

    def _item_parameters_from_semantic(self, semantic: Tensor) -> dict[str, Tensor]:
        if self.prediction_head != "2pl":
            raise ValueError("The MLP control does not define IRT item parameters")
        return {"b": 3.0 * torch.tanh(self.difficulty_head(semantic)),
                "a": F.softplus(self.discrimination_head(semantic)) + 0.1}

    def item_parameters(self, concept_emb: Tensor,
                        question_emb: Tensor) -> dict[str, Tensor]:
        """Compute b and a for arbitrary item-leading dimensions, including [N,D]."""
        return self._item_parameters_from_semantic(
            self.encode_items(concept_emb, question_emb))

    def ability(self, concept_emb: Tensor, question_emb: Tensor,
                responses: Tensor) -> Tensor:
        """Return theta_t = g(H_<t), independent of the current target item."""
        if self.prediction_head != "2pl":
            raise ValueError("The MLP control does not define a scalar IRT ability")
        semantic = self.encode_items(concept_emb, question_emb)
        return self.ability_head(self._history_before(semantic, responses))

    def forward_details(self, concept_emb: Tensor, question_emb: Tensor,
                        responses: Tensor) -> dict[str, Tensor]:
        semantic = self.encode_items(concept_emb, question_emb)
        before = self._history_before(semantic, responses)
        if self.prediction_head == "mlp":
            logits = self.response_head(torch.cat((before, semantic), dim=-1))
            return {"probabilities": torch.sigmoid(logits), "logits": logits}
        parameters = self._item_parameters_from_semantic(semantic)
        theta = self.ability_head(before)
        logits = parameters["a"] * (theta - parameters["b"])
        return {"probabilities": torch.sigmoid(logits), "logits": logits,
                "theta": theta, **parameters}

    def forward(self, concept_emb: Tensor, question_emb: Tensor,
                responses: Tensor) -> Tensor:
        return self.forward_details(concept_emb, question_emb,
                                    responses)["probabilities"]

    def regularization(self, concept_emb: Tensor, question_emb: Tensor) -> Tensor:
        """Weak location/scale anchor; runner chooses its coefficient.

        Pass UNIQUE TRAIN ITEMS here. This only discourages drift of the mean
        difficulty and geometric-mean discrimination; it does not establish
        complete IRT identifiability or calibrated psychological measurements.
        """
        if self.prediction_head != "2pl":
            return self.concept_projection[0].weight.new_zeros(())
        parameters = self.item_parameters(concept_emb, question_emb)
        return parameters["b"].mean().square() + parameters["a"].log().mean().square()


class OriginalTextAKT(nn.Module):
    """Original local TextAKT3PL, with its computation and hyperparameters intact.

    Its module-global device is synchronized with inputs to allow CPU checks on
    hosts that also have a GPU. This changes tensor placement, not the equations.
    """

    def __init__(self, input_dim: int, model_dim: int, seq_len: int) -> None:
        super().__init__()
        if model_dim % 8:
            raise ValueError("The original eight-head AKT needs model_dim divisible by 8")
        module = importlib.import_module("model.my_model")
        self.model = module.TextAKT3PL(
            input_dim=input_dim, model_dim=model_dim, base_scale=0.2,
            n_block=2, n_heads=8, dropout=0.0, kq_same=1, seq_len=seq_len)

    def forward(self, concept_emb: Tensor, question_emb: Tensor,
                responses: Tensor) -> Tensor:
        module = importlib.import_module("model.my_model")
        module.device = question_emb.device
        if responses.ndim == 2:
            responses = responses.unsqueeze(-1)
        return self.model(concept_emb.detach(), question_emb.detach(), responses.detach())


def build_model(variant: str, input_dim: int = 256, model_dim: int = 128,
                seq_len: int = 201, dropout: float = 0.1) -> nn.Module:
    """Build original, semantic_mlp, decoupled, or aligned experiment models."""
    if variant == "original":
        return OriginalTextAKT(input_dim, model_dim, seq_len)
    if variant not in {"semantic_mlp", "decoupled", "aligned"}:
        raise ValueError(f"Unknown model variant: {variant}")
    return SemanticHistoryModel(input_dim, model_dim, dropout,
                                prediction_head="mlp" if variant == "semantic_mlp" else "2pl")
