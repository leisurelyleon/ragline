"""Deterministic, local providers requiring no network or API key.

These are real implementations, not mocks: the embedder maps text to vectors via
feature hashing, and the generator produces a grounded, templated answer from the
retrieved context. They make the entire pipeline, its tests, and the evaluation
harness fully reproducible offline.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import numpy.typing as npt

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """Embeds text using the feature-hashing trick.

    Each token is hashed (SHA-256, for cross-run determinism) to a bucket in a
    fixed-dimension vector; counts accumulate and the vector is L2-normalized.
    Dependency-free, deterministic, and still placing texts that share vocabulary
    near one another so retrieval is meaningful.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False) % self._dimension

    def embed(self, texts: list[str]) -> npt.NDArray[np.float64]:
        vectors = np.zeros((len(texts), self._dimension), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                vectors[row, self._bucket(token)] += 1.0
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0.0:
                vectors[row] /= norm
        return vectors


class TemplateGenerator:
    """Generates a grounded answer by stitching together retrieved context.

    Deterministic: the answer quotes the retrieved passages directly, so the
    pipeline produces stable, inspectable output with no model call.
    """

    def __init__(self, max_passages: int = 3) -> None:
        if max_passages <= 0:
            raise ValueError("max_passages must be positive")
        self._max_passages = max_passages

    def generate(self, question: str, context: list[str]) -> str:
        if not context:
            return f"No relevant context was found to answer: {question}"
        used = context[: self._max_passages]
        joined = " ".join(passage.strip() for passage in used)
        return f"Based on the retrieved context: {joined}"
