"""Embedding abstraction for ablation variants."""

from abc import ABC

from torch import Tensor, nn


class BaseEmbedding(nn.Module, ABC):
    """Base embedding module."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed token ids into dense vectors."""
        raise NotImplementedError
