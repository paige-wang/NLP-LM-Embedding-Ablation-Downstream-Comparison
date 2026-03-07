"""Vocabulary with special tokens and JSON serialization."""

from collections import Counter
import json
from pathlib import Path


class Vocabulary:
    """Token-index vocabulary with fixed special-token ids."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

    def __init__(self, max_size: int = 20000) -> None:
        self.max_size = max_size
        self.token2id: dict[str, int] = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.id2token: dict[int, str] = {idx: token for token, idx in self.token2id.items()}

    @property
    def pad_id(self) -> int:
        return self.token2id[self.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2id[self.UNK]

    @property
    def bos_id(self) -> int:
        return self.token2id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token2id[self.EOS]

    def __len__(self) -> int:
        return len(self.token2id)

    def build(self, token_sequences: list[list[str]]) -> None:
        """Build vocab from tokenized sequences."""
        counter = Counter(token for seq in token_sequences for token in seq)
        capacity = max(self.max_size, 0)
        for token, _ in counter.most_common(capacity):
            if token in self.token2id:
                continue
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    def encode_tokens(self, tokens: list[str], add_bos_eos: bool = False) -> list[int]:
        """Encode token list into ids."""
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode_ids(self, ids: list[int], skip_special: bool = True) -> list[str]:
        """Decode ids into tokens."""
        tokens: list[str] = []
        for idx in ids:
            token = self.id2token.get(idx, self.UNK)
            if skip_special and token in self.SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def save(self, path: str) -> None:
        """Save vocabulary JSON with UTF-8 encoding."""
        payload = {
            "max_size": self.max_size,
            "token2id": self.token2id,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vocab = cls(max_size=int(payload["max_size"]))
        token2id = {str(k): int(v) for k, v in payload["token2id"].items()}
        vocab.token2id = token2id
        vocab.id2token = {idx: token for token, idx in token2id.items()}
        return vocab
