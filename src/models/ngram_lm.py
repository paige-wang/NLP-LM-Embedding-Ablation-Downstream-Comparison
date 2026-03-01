"""Sparse trigram LM with Laplace smoothing."""

from collections import Counter, defaultdict
import math
from typing import Iterable

import numpy as np


class NGramLanguageModel:
    """Count-based n-gram model using sparse nested dictionaries."""

    def __init__(self, vocab_size: int, n: int = 3, smoothing: float = 1.0) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.vocab_size = vocab_size
        self.n = n
        self.smoothing = smoothing
        self.context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.context_totals: Counter[tuple[int, ...]] = Counter()

    def fit(self, token_ids: Iterable[int]) -> None:
        """Fit counts from a token-id stream."""
        ids = list(token_ids)
        for idx in range(self.n - 1, len(ids)):
            context = tuple(ids[idx - (self.n - 1) : idx])
            target = ids[idx]
            self.context_counts[context][target] += 1
            self.context_totals[context] += 1

    def prob(self, context: tuple[int, ...], token_id: int) -> float:
        """Compute Laplace-smoothed conditional probability."""
        count = self.context_counts[context][token_id]
        total = self.context_totals[context]
        numerator = count + self.smoothing
        denominator = total + self.smoothing * self.vocab_size
        return numerator / denominator

    def perplexity(self, token_ids: Iterable[int]) -> float:
        """Compute perplexity of an id stream."""
        ids = list(token_ids)
        if len(ids) <= self.n - 1:
            return float("inf")
        nll = 0.0
        steps = 0
        for idx in range(self.n - 1, len(ids)):
            context = tuple(ids[idx - (self.n - 1) : idx])
            target = ids[idx]
            probability = self.prob(context, target)
            nll += -math.log(max(probability, 1e-12))
            steps += 1
        return float(np.exp(nll / max(steps, 1)))

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        """Greedily generate continuation from the prompt."""
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            if len(generated) < self.n - 1:
                break
            context = tuple(generated[-(self.n - 1) :])
            candidate_counts = self.context_counts.get(context)
            if not candidate_counts:
                generated.append(1)
                break
            next_id, _ = candidate_counts.most_common(1)[0]
            generated.append(next_id)
        return generated
