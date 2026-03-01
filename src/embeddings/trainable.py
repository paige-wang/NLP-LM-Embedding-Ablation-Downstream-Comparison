"""Trainable embedding module."""

from torch import Tensor, nn

from src.embeddings.base import BaseEmbedding


class TrainableEmbedding(BaseEmbedding):
    """Standard ``nn.Embedding`` updated during LM training."""

    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int = 0) -> None:
        super().__init__(embedding_dim=embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: [B, T]
        # return:    [B, T, D]
        return self.embedding(token_ids)
