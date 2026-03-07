"""Base interfaces for neural language models."""

from abc import ABC, abstractmethod
import math

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class BaseLanguageModel(nn.Module, ABC):
    """Abstract base class for neural language models."""

    def __init__(self, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    @abstractmethod
    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute logits from token ids."""

    @abstractmethod
    def encode(self, input_ids: Tensor) -> Tensor:
        """Return sequence representations for downstream usage."""

    def compute_loss(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Compute cross-entropy loss for LM training."""
        # input_ids:  [B, T]
        # target_ids: [B, T]
        logits = self.forward(input_ids)  # [B, T, V]
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        return loss

    @torch.no_grad()
    def perplexity(self, dataloader: DataLoader, device: torch.device) -> float:
        """Evaluate perplexity while ignoring ``<pad>`` tokens."""
        self.eval()
        total_loss = 0.0
        total_tokens = 0
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = self.forward(input_ids)  # [B, T, V]
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            loss = self.criterion(flat_logits, flat_targets)
            valid_tokens = int((flat_targets != self.pad_id).sum().item())
            total_loss += float(loss.item()) * max(valid_tokens, 1)
            total_tokens += valid_tokens

        if total_tokens == 0:
            return float("inf")
        avg_nll = total_loss / total_tokens
        return math.exp(avg_nll)
