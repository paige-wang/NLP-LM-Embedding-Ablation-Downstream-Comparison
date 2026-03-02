"""Downstream intent classifier on top of LM encoder representations."""

import torch
from torch import Tensor, nn

from src.models.base_lm import BaseLanguageModel


class IntentClassifier(nn.Module):
    """Last-token pooling + linear classifier for SNIPS intent prediction."""

    def __init__(self, lm: BaseLanguageModel, embed_dim: int = 300, num_classes: int = 7) -> None:
        super().__init__()
        self.lm = lm
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        hidden = self.lm.encode(input_ids)  # [B, T, D]
        # Last-token pooling: pick the representation at the last non-pad token per sequence.
        pad_id = getattr(self.lm, "pad_id", 0)
        mask = (input_ids != pad_id)  # [B, T]
        lengths = mask.sum(dim=1).clamp(min=1)  # [B]
        # index of last real token for each batch element
        idx = (lengths - 1).to(torch.long)  # [B]
        # prepare index for gather: [B, 1, D]
        idx_exp = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden.size(2)).to(hidden.device)
        pooled = hidden.gather(1, idx_exp).squeeze(1)  # [B, D]
        assert pooled.ndim == 2, f"Expected 2D pooled tensor, got {pooled.ndim}D"
        assert pooled.shape[-1] == self.classifier.in_features, (
            f"Expected feature dim {self.classifier.in_features}, got {pooled.shape[-1]}"
        )
        logits = self.classifier(pooled)  # [B, C]
        return logits
