"""Custom causal Transformer language model."""

import math

import torch
from torch import Tensor, nn

from src.embeddings.base import BaseEmbedding
from src.models.base_lm import BaseLanguageModel


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        # x:           [B, T, D]
        # causal_mask: [T, T] bool
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = self.k_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        v = self.v_proj(x).view(bsz, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, T, T]
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ v  # [B, H, T, Hd]
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)  # [B, T, D]
        output = self.out_proj(context)  # [B, T, D]
        return output


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        # x:           [B, T, D]
        # causal_mask: [T, T] bool
        x = x + self.dropout1(self.attn(self.norm1(x), causal_mask))  # [B, T, D]
        x = x + self.dropout2(self.ffn(self.norm2(x)))  # [B, T, D]
        return x


class TransformerLanguageModel(BaseLanguageModel):
    """Transformer LM with causal masking and positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding: BaseEmbedding,
        d_model: int = 300,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1200,
        dropout: float = 0.5,
        pad_id: int = 0,
        tie_weights: bool = False,
    ) -> None:
        super().__init__(pad_id=pad_id)
        if embedding.embedding_dim != d_model:
            raise ValueError("Embedding dimension must equal d_model for Transformer")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights and hasattr(self.embedding, "weight") and self.embedding.weight.shape[0] == vocab_size:
            self.lm_head.weight = self.embedding.weight

    def _positional_encoding(self, seq_len: int, device: torch.device) -> Tensor:
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )  # [D/2]
        pe = torch.zeros(seq_len, self.d_model, device=device, dtype=torch.float32)  # [T, D]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, T, D]

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()  # [T, T]
        return mask

    def encode(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        bsz, seq_len = input_ids.shape
        x = self.embedding(input_ids)  # [B, T, D]
        pos = self._positional_encoding(seq_len, input_ids.device)  # [1, T, D]
        x = self.dropout(x + pos)  # [B, T, D]
        causal_mask = self._build_causal_mask(seq_len, input_ids.device)  # [T, T]
        for block in self.blocks:
            x = block(x, causal_mask)  # [B, T, D]
        encoded = self.final_norm(x)  # [B, T, D]
        assert encoded.ndim == 3, f"Expected [B,T,D], got {encoded.shape}"
        assert encoded.shape[0] == bsz and encoded.shape[1] == seq_len
        return encoded

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]
        encoded = self.encode(input_ids)  # [B, T, D]
        logits = self.lm_head(encoded)  # [B, T, V]
        return logits
