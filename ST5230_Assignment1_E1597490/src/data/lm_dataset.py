"""Language modeling dataset with left-shifted targets."""

from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class LMDataset(Dataset[tuple[Tensor, Tensor]]):
    """Create fixed-length LM samples from a token-id stream."""

    def __init__(self, token_ids: Sequence[int] | np.ndarray, seq_len: int = 64, stride: int = 1) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        self.seq_len = seq_len
        self.stride = stride
        self.num_samples = max((len(self.token_ids) - seq_len - 1) // stride + 1, 0)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.stride
        end = start + self.seq_len
        # input_ids:  [T]
        input_ids = torch.as_tensor(self.token_ids[start:end], dtype=torch.long)
        # target_ids: [T] (left-shifted by one token)
        target_ids = torch.as_tensor(self.token_ids[start + 1 : end + 1], dtype=torch.long)
        return input_ids, target_ids
