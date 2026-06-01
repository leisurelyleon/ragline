"""Tests for document chunking."""

from __future__ import annotations

import pytest

from ragline.chunking import chunk_document
from ragline.document import Document


def test_empty_document_yields_no_chunks() -> None:
    doc = Document(doc_id="empty", text="")
    assert chunk_document(doc) == []


def test_short_document_is_single_chunk() -> None:
    doc = Document(doc_id="short", text="hello world")
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start == 0
    assert chunks[0].end == len("hello world")


def test_chunks_cover_and_overlap() -> None:
    text = "abcdefghij"  # 10 chars
    doc = Document(doc_id="d", text=text)
    chunks = chunk_document(doc, chunk_size=4, overlap=2)
    # step = 2: starts at 0,2,4,6,8
    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij", "ij"]
    # consecutive chunks share `overlap` characters
    assert chunks[0].text[-2:] == chunks[1].text[:2]


def test_chunk_ids_are_unique_and_ordered() -> None:
    doc = Document(doc_id="doc", text="abcdefghij")
    chunks = chunk_document(doc, chunk_size=4, overlap=1)
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(ids, key=lambda s: int(s.split(":")[1]))
    assert len(set(ids)) == len(ids)


def test_length_property() -> None:
    doc = Document(doc_id="d", text="abcdef")
    chunks = chunk_document(doc, chunk_size=4, overlap=1)
    assert chunks[0].length == len(chunks[0].text)


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 15)],
)
def test_invalid_parameters_raise(chunk_size: int, overlap: int) -> None:
    doc = Document(doc_id="d", text="abc")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
