"""Document and chunk data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A source document with an identifier and text content."""

    doc_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document; the unit of retrieval."""

    chunk_id: str
    doc_id: str
    text: str
    start: int  # 0-based character offset in the source document
    end: int  # exclusive end offset

    @property
    def length(self) -> int:
        return self.end - self.start
