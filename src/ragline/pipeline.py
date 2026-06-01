"""The end-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ragline.chunking import chunk_document
from ragline.document import Chunk, Document
from ragline.providers.base import Embedder, Generator
from ragline.vector_store import ScoredChunk, VectorStore


@dataclass(frozen=True)
class QueryResult:
    """The result of a query: the generated answer and the chunks it drew on."""

    question: str
    answer: str
    sources: list[ScoredChunk]


class Pipeline:
    """Chunks and embeds documents, then answers questions over them."""

    def __init__(
        self,
        embedder: Embedder,
        generator: Generator,
        chunk_size: int = 200,
        overlap: int = 40,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._top_k = top_k
        self._store = VectorStore(embedder.dimension)

    @property
    def store(self) -> VectorStore:
        return self._store

    def ingest(self, documents: list[Document]) -> int:
        """Chunk, embed, and store a batch of documents. Returns chunks added."""
        all_chunks: list[Chunk] = []
        for document in documents:
            all_chunks.extend(chunk_document(document, self._chunk_size, self._overlap))
        if not all_chunks:
            return 0
        embeddings = self._embedder.embed([chunk.text for chunk in all_chunks])
        self._store.add(all_chunks, embeddings)
        return len(all_chunks)

    def retrieve(self, question: str) -> list[ScoredChunk]:
        """Retrieve the most relevant chunks for a question."""
        query_embedding = self._embedder.embed([question])
        return self._store.search(query_embedding[0], self._top_k)

    def query(self, question: str) -> QueryResult:
        """Answer a question using retrieved context."""
        sources = self.retrieve(question)
        context = [scored.chunk.text for scored in sources]
        answer = self._generator.generate(question, context)
        return QueryResult(question=question, answer=answer, sources=sources)
