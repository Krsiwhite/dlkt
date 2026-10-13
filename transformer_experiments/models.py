"""Minimal item-head ablations of the original local Transformer/AKT model.

All variants initialize the same TextAKT3PL parameters in the same order. The
static and aligned variants only feed the three item heads their projected
semantic inputs before positions and contextual attention. The original
contextual feature, interaction gates, position embeddings, AKT blocks and 3PL
prediction equation remain in use. Aligned is the same model as static; its
training-only teacher loss is applied by the runner.

This is a static-item-parameter ablation, not full psychometric disentanglement:
the AKT ability is still target-conditioned, and the original item-dependent
interaction gates remain. No trainable question or student ID lookup is added.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.models import OriginalTextAKT
from model.my_model import SemanticIRTEncoder


class StaticSemanticIRTEncoder(SemanticIRTEncoder):
    """Use unchanged original modules with a static input to the item heads.

The class is installed on an already initialized original extractor. It has no
new attributes or parameters, so the initialization, state_dict and checkpoint
layout are identical across all three variants.
"""

    def _semantic_features(self, concept_emb: Tensor,
                           question_emb: Tensor) -> tuple[Tensor, Tensor]:
        if concept_emb.shape != question_emb.shape:
            raise ValueError("Concept and question vectors must have equal shapes")
        c_vec = self.concept_proj(concept_emb.detach())
        q_feat = self.q_shared(question_emb.detach())
        return c_vec + q_feat, q_feat

    def _item_heads(self, semantic: Tensor) -> dict[str, Tensor]:
        return {"b": 3.0 * torch.tanh(self.head_mu(semantic)),
                "a": F.softplus(self.head_alpha(semantic)) + 0.1,
                "guess": torch.sigmoid(self.head_guess(semantic))}

    def item_parameters(self, concept_emb: Tensor,
                        question_emb: Tensor) -> dict[str, Tensor]:
        """Return static b, a and guess for [N,D] or [B,L,D] raw vectors.

The original projections and heads have zero dropout and no batch-dependent
normalization. Therefore this path is deterministic in train mode as well as
eval mode, independent of surrounding items, position and response history.
"""
        semantic, _ = self._semantic_features(concept_emb, question_emb)
        return self._item_heads(semantic)

    def forward(self, concept_emb: Tensor, question_emb: Tensor):
        semantic, q_feat = self._semantic_features(concept_emb, question_emb)
        # Deliberately do not use +=: that would modify the static head input
        # through its alias and silently reintroduce absolute position.
        contextual = semantic + self.pos_encoder(semantic)
        f_vec = self.mix_diff(mask=1, query=contextual, key=contextual,
                             values=contextual)
        parameters = self._item_heads(semantic)
        return (f_vec, q_feat, parameters["b"], parameters["a"],
                parameters["guess"])


class StaticTextAKT(OriginalTextAKT):
    """Original TextAKT3PL with only the item-head input path made static."""

    def __init__(self, input_dim: int, model_dim: int, seq_len: int) -> None:
        super().__init__(input_dim, model_dim, seq_len)
        # Both classes have the same nn.Module layout and inherited initializer.
        # Changing only forward avoids initializing replacement modules, which
        # would otherwise change the RNG stream and the matched-seed controls.
        self.model.extractor.__class__ = StaticSemanticIRTEncoder

    def item_parameters(self, concept_emb: Tensor,
                        question_emb: Tensor) -> dict[str, Tensor]:
        return self.model.extractor.item_parameters(concept_emb, question_emb)


def build_model(variant: str, input_dim: int = 256, model_dim: int = 128,
                seq_len: int = 201) -> nn.Module:
    """Build original, static or aligned with identical initial parameters.

No anchor regularizer is defined here. The sole difference between static and
aligned training is the external difficulty-alignment loss.
"""
    if variant == "original":
        return OriginalTextAKT(input_dim, model_dim, seq_len)
    if variant in {"static", "aligned"}:
        return StaticTextAKT(input_dim, model_dim, seq_len)
    raise ValueError(f"Unknown Transformer variant: {variant}")
