"""Retrieval and faithfulness metrics. Pure functions over IDs and text."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found within the top-k retrieved."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal of the rank of the first relevant item (0.0 if none found)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def faithfulness(answer: str, context: list[str]) -> float:
    """A lightweight lexical proxy for grounding: the fraction of the answer's
    content words that appear in the retrieved context.

    This is a heuristic, not a model-based judgment. It rewards answers whose
    vocabulary is supported by the context and penalizes unsupported content.
    """
    answer_tokens = set(_TOKEN_RE.findall(answer.lower()))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for passage in context:
        context_tokens.update(_TOKEN_RE.findall(passage.lower()))
    return len(answer_tokens & context_tokens) / len(answer_tokens)
