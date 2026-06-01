"""An in-memory vector store with cosine-similarity top-k retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ragline.document import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk paired with its similarity score against a query."""

    chunk: Chunk
    score: float


class VectorStore:
    """Stores chunk embeddings and retrieves the most similar chunks.

    Embeddings are assumed L2-normalized, so cosine similarity reduces to a dot
    product. Dimensionality is validated on insert and on query.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._chunks: list[Chunk] = []
        self._matrix: npt.NDArray[np.float64] = np.zeros((0, dimension), dtype=np.float64)

    @property
    def dimension(self) -> int:
        return self._dimension

    def __len__(self) -> int:
        return len(self._chunks)

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def add(self, chunks: list[Chunk], embeddings: npt.NDArray[np.float64]) -> None:
        """Add chunks and their corresponding embedding rows."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"embeddings must have shape (n, {self._dimension}); got {embeddings.shape}"
            )
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("number of chunks must match number of embedding rows")
        self._chunks.extend(chunks)
        self._matrix = np.vstack([self._matrix, embeddings]).astype(np.float64)

    def search(self, query: npt.NDArray[np.float64], top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks most similar to the query vector."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.is_empty():
            return []

        vector = query.reshape(-1)
        if vector.shape[0] != self._dimension:
            raise ValueError(
                f"query dimension {vector.shape[0]} does not match "
                f"store dimension {self._dimension}"
            )

        scores: npt.NDArray[np.float64] = (self._matrix @ vector).astype(np.float64)
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            ScoredChunk(chunk=self._chunks[int(i)], score=float(scores[int(i)]))
            for i in top_indices
        ]
