"""Fixed pretrained GloVe embedding with random OOV initialization."""

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from src.data.vocabulary import Vocabulary
from src.embeddings.base import BaseEmbedding


class FixedPretrainedEmbedding(BaseEmbedding):
    """GloVe-based frozen embedding."""

    def __init__(self, weight: Tensor, pad_id: int = 0) -> None:
        super().__init__(embedding_dim=weight.shape[1])
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=pad_id)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: [B, T]
        # return:    [B, T, D]
        return self.embedding(token_ids)

    @classmethod
    def from_glove(
        cls,
        vocab: Vocabulary,
        glove_path: str,
        embed_dim: int,
        pad_id: int = 0,
    ) -> tuple["FixedPretrainedEmbedding", float]:
        """Load GloVe vectors and align them to LM vocabulary."""
        path = Path(glove_path)
        glove_map: dict[str, np.ndarray] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip().split(" ")
                    if len(parts) != embed_dim + 1:
                        continue
                    token = parts[0]
                    vector = np.asarray(parts[1:], dtype=np.float32)
                    glove_map[token] = vector

        weight = np.random.randn(len(vocab), embed_dim).astype(np.float32) * 0.01
        hits = 0
        for token, idx in vocab.token2id.items():
            if token in glove_map:
                weight[idx] = glove_map[token]
                hits += 1
        coverage = hits / max(len(vocab), 1)
        return cls(weight=torch.tensor(weight), pad_id=pad_id), coverage
