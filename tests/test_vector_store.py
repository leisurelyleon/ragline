"""Tests for the NumPy vector store."""

from __future__ import annotations

import numpy as np
import pytest

from ragline.document import Chunk
from ragline.vector_store import VectorStore


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="d", text=chunk_id, start=0, end=1)


def _unit(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def test_empty_store() -> None:
    store = VectorStore(dimension=3)
    assert store.is_empty()
    assert len(store) == 0
    assert store.search(np.array([1.0, 0.0, 0.0]), top_k=3) == []


def test_add_and_search_returns_most_similar() -> None:
    store = VectorStore(dimension=2)
    store.add(
        [_chunk("east"), _chunk("north")],
        np.array([_unit([1.0, 0.0]), _unit([0.0, 1.0])]),
    )
    results = store.search(_unit([0.9, 0.1]), top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "east"


def test_search_orders_by_score_descending() -> None:
    store = VectorStore(dimension=2)
    store.add(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        np.array([_unit([1.0, 0.0]), _unit([0.7, 0.7]), _unit([0.0, 1.0])]),
    )
    results = store.search(_unit([1.0, 0.0]), top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk.chunk_id == "a"


def test_top_k_caps_at_store_size() -> None:
    store = VectorStore(dimension=2)
    store.add([_chunk("only")], np.array([_unit([1.0, 0.0])]))
    results = store.search(_unit([1.0, 0.0]), top_k=10)
    assert len(results) == 1


def test_add_rejects_wrong_dimension() -> None:
    store = VectorStore(dimension=3)
    with pytest.raises(ValueError):
        store.add([_chunk("x")], np.array([[1.0, 0.0]]))  # 2-dim, store is 3


def test_add_rejects_count_mismatch() -> None:
    store = VectorStore(dimension=2)
    with pytest.raises(ValueError):
        store.add([_chunk("x"), _chunk("y")], np.array([_unit([1.0, 0.0])]))  # 2 chunks, 1 row


def test_search_rejects_wrong_query_dimension() -> None:
    store = VectorStore(dimension=3)
    store.add([_chunk("x")], np.array([_unit([1.0, 0.0, 0.0])]))
    with pytest.raises(ValueError):
        store.search(np.array([1.0, 0.0]), top_k=1)  # 2-dim query


def test_search_rejects_bad_top_k() -> None:
    store = VectorStore(dimension=2)
    store.add([_chunk("x")], np.array([_unit([1.0, 0.0])]))
    with pytest.raises(ValueError):
        store.search(_unit([1.0, 0.0]), top_k=0)
