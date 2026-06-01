"""Optional OpenAI-backed providers.

Imported only when the OpenAI provider is explicitly selected. Requires the
optional `openai` dependency (`pip install "ragline[openai]"`) and an
OPENAI_API_KEY in the environment. Nothing in the default pipeline, tests, or
evaluation harness imports this module.
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - only hit without the extra
    raise ImportError(
        'The OpenAI provider requires the optional dependency. Install with: '
        'pip install "ragline[openai]"'
    ) from exc


class OpenAIEmbedder:
    """Embeds text using OpenAI's embedding API."""

    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536) -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> npt.NDArray[np.float64]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return np.array([item.embedding for item in response.data], dtype=np.float64)


class OpenAIGenerator:
    """Generates answers using an OpenAI chat model."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def generate(self, question: str, context: list[str]) -> str:
        joined = "\n\n".join(context)
        prompt = (
            "Answer the question using only the provided context. "
            "If the context is insufficient, say so.\n\n"
            f"Context:\n{joined}\n\nQuestion: {question}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        return answer if answer is not None else ""
