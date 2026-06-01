"""Tests for the retrieval and faithfulness metrics."""

from __future__ import annotations

import pytest

from ragline.evaluation.metrics import (
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}  # z is never retrieved
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)


def test_recall_with_no_relevant_is_zero() -> None:
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_mean_reciprocal_rank() -> None:
    assert mean_reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert mean_reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_faithfulness_full_overlap() -> None:
    answer = "solar energy"
    context = ["solar energy comes from the sun"]
    assert faithfulness(answer, context) == 1.0


def test_faithfulness_partial_overlap() -> None:
    answer = "solar banana"
    context = ["solar energy from the sun"]
    # "solar" is grounded, "banana" is not -> 0.5
    assert faithfulness(answer, context) == pytest.approx(0.5)


def test_faithfulness_empty_answer_is_zero() -> None:
    assert faithfulness("", ["some context"]) == 0.0


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=-1)
