"""Fixed self-trained Word2Vec embedding."""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from src.data.vocabulary import Vocabulary
from src.embeddings.base import BaseEmbedding

try:
    from gensim.models import KeyedVectors, Word2Vec
except Exception:  # pragma: no cover - optional import at runtime
    KeyedVectors = None
    Word2Vec = None


class FixedSelfTrainedEmbedding(BaseEmbedding):
    """Word2Vec-based embedding frozen during LM training."""

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
    def from_corpus(
        cls,
        vocab: Vocabulary,
        token_sequences: Iterable[list[str]],
        embed_dim: int,
        save_path: str,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
        workers: int = 1,
        pad_id: int = 0,
    ) -> "FixedSelfTrainedEmbedding":
        """Train/load Word2Vec and align vectors strictly to LM vocabulary ids."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if KeyedVectors is not None and path.exists():
            kv = KeyedVectors.load(str(path), mmap="r")
        elif Word2Vec is not None:
            sentences = list(token_sequences)
            if not sentences:
                raise ValueError("fixed_self requires non-empty token sequences for Word2Vec training.")
            model = Word2Vec(
                vector_size=embed_dim,
                window=window,
                min_count=min_count,
                workers=workers,
            )
            model.build_vocab(sentences)
            model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
            kv = model.wv
            kv.save(str(path))
        else:
            kv = None

        weight = np.random.randn(len(vocab), embed_dim).astype(np.float32) * 0.01
        for token, idx in vocab.token2id.items():
            if kv is not None and token in kv:
                weight[idx] = kv[token]
        return cls(weight=torch.tensor(weight), pad_id=pad_id)
