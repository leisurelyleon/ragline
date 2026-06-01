"""Split documents into overlapping chunks for retrieval. Pure logic."""

from __future__ import annotations

from ragline.document import Chunk, Document


def chunk_document(
    document: Document,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[Chunk]:
    """Split a document into overlapping character chunks.

    Args:
        document: the source document.
        chunk_size: maximum characters per chunk.
        overlap: characters shared between consecutive chunks.

    Returns:
        An ordered list of chunks covering the document.

    Raises:
        ValueError: if chunk_size <= 0, overlap < 0, or overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.text
    if not text:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{index}",
                doc_id=document.doc_id,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        index += 1
        if end == len(text):
            break
        start += step

    return chunks
