"""Provider interfaces: the seam between the pipeline and concrete backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Embedder(Protocol):
    """Turns text into fixed-dimension embedding vectors."""

    @property
    def dimension(self) -> int:
        """The dimensionality of the vectors this embedder produces."""
        ...

    def embed(self, texts: list[str]) -> npt.NDArray[np.float64]:
        """Embed a batch of texts into a (len(texts), dimension) array."""
        ...


class Generator(Protocol):
    """Generates an answer given a question and retrieved context."""

    def generate(self, question: str, context: list[str]) -> str:
        """Produce an answer grounded in the provided context passages."""
        ...
