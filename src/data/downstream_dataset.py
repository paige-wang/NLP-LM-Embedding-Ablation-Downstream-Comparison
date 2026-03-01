"""SNIPS downstream dataset helpers."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.tokenizer import SimpleTokenizer
from src.data.vocabulary import Vocabulary


SNIPS_INTENTS = [
    "AddToPlaylist",
    "BookRestaurant",
    "GetWeather",
    "PlayMusic",
    "RateBook",
    "SearchCreativeWork",
    "SearchScreeningEvent",
]


@dataclass
class SNIPSExample:
    """One downstream sample."""

    text: str
    label: int


class SNIPSDataset(Dataset[tuple[Tensor, Tensor]]):
    """SNIPS dataset encoded with LM vocabulary."""

    def __init__(self, examples: list[SNIPSExample], vocab: Vocabulary, seq_len: int = 32) -> None:
        self.examples = examples
        self.vocab = vocab
        self.seq_len = seq_len
        self.tokenizer = SimpleTokenizer()

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, text: str) -> list[int]:
        tokens = self.tokenizer.tokenize(text)
        ids = self.vocab.encode_tokens(tokens, add_bos_eos=True)
        if len(ids) >= self.seq_len:
            return ids[: self.seq_len]
        pad_needed = self.seq_len - len(ids)
        return ids + [self.vocab.pad_id] * pad_needed

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        example = self.examples[idx]
        # input_ids: [T]
        input_ids = torch.tensor(self._encode(example.text), dtype=torch.long)
        # label: []
        label = torch.tensor(example.label, dtype=torch.long)
        return input_ids, label


def parse_snips_split(dataset_split: Any) -> list[SNIPSExample]:
    """Convert HuggingFace SNIPS split into normalized examples."""
    examples: list[SNIPSExample] = []
    label2id = {name: idx for idx, name in enumerate(SNIPS_INTENTS)}

    for row in dataset_split:
        text = row.get("text", "").strip()
        label_value = row.get("intent")
        if isinstance(label_value, int):
            label = int(label_value)
        else:
            label = label2id.get(str(label_value), 0)
        examples.append(SNIPSExample(text=text, label=label))
    return examples
