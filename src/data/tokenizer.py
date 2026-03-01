"""Simple whitespace tokenizer for WikiText/SNIPS preprocessing."""

import re
from typing import Iterable


class SimpleTokenizer:
    """Tokenizer that lowercases and splits by whitespace."""

    _space_re = re.compile(r"\s+")

    def tokenize(self, text: str) -> list[str]:
        """Tokenize one text string into lowercase tokens."""
        normalized = self._space_re.sub(" ", text.strip().lower())
        if not normalized:
            return []
        return normalized.split(" ")

    def batch_tokenize(self, texts: Iterable[str]) -> list[list[str]]:
        """Tokenize many strings."""
        return [self.tokenize(text) for text in texts]

    @staticmethod
    def is_valid_wikitext_line(line: str) -> bool:
        """Filter short lines and section headers per blueprint constraints."""
        stripped = line.strip()
        if len(stripped) < 10:
            return False
        if stripped.startswith("=") and stripped.endswith("="):
            return False
        return True
