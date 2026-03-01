"""Downstream intent classifier on top of LM encoder representations."""

from torch import Tensor, nn

from src.models.base_lm import BaseLanguageModel


class IntentClassifier(nn.Module):
    """Mean-pooling + linear classifier for SNIPS intent prediction."""

    def __init__(self, lm: BaseLanguageModel, embed_dim: int = 300, num_classes: int = 7) -> None:
        super().__init__()
        self.lm = lm
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        hidden = self.lm.encode(input_ids)  # [B, T, D]
        pooled = hidden.mean(dim=1)  # [B, D]
        assert pooled.ndim == 2, f"Expected 2D pooled tensor, got {pooled.ndim}D"
        assert pooled.shape[-1] == self.classifier.in_features, (
            f"Expected feature dim {self.classifier.in_features}, got {pooled.shape[-1]}"
        )
        logits = self.classifier(pooled)  # [B, C]
        return logits
