"""LSTM language model."""

from torch import Tensor, nn

from src.embeddings.base import BaseEmbedding
from src.models.base_lm import BaseLanguageModel


class LSTMLanguageModel(BaseLanguageModel):
    """Two-layer LSTM LM with pluggable embedding."""

    def __init__(
        self,
        vocab_size: int,
        embedding: BaseEmbedding,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__(pad_id=pad_id)
        self.vocab_size = vocab_size
        self.embedding = embedding
        self.lstm = nn.LSTM(
            input_size=embedding.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.encoder_proj = nn.Linear(hidden_size, embedding.embedding_dim)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        x = self.embedding(input_ids)  # [B, T, D]
        hidden_states, _ = self.lstm(x)  # [B, T, H]
        logits = self.lm_head(hidden_states)  # [B, T, V]
        return logits

    def encode(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        x = self.embedding(input_ids)  # [B, T, D]
        hidden_states, _ = self.lstm(x)  # [B, T, H]
        encoded = self.encoder_proj(hidden_states)  # [B, T, D]
        return encoded
