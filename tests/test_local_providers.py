"""Tests for the deterministic local providers."""

from __future__ import annotations

import numpy as np
import pytest

from ragline.providers.local import HashEmbedder, TemplateGenerator


def test_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dimension=64)
    a = embedder.embed(["the quick brown fox"])
    b = embedder.embed(["the quick brown fox"])
    assert np.array_equal(a, b)


def test_embedder_output_shape_and_normalization() -> None:
    embedder = HashEmbedder(dimension=128)
    vectors = embedder.embed(["alpha beta", "gamma"])
    assert vectors.shape == (2, 128)
    # Each non-empty row is L2-normalized (unit length).
    for row in vectors:
        assert np.isclose(np.linalg.norm(row), 1.0)


def test_embedder_empty_text_is_zero_vector() -> None:
    embedder = HashEmbedder(dimension=32)
    vectors = embedder.embed([""])
    assert np.allclose(vectors[0], 0.0)


def test_similar_texts_score_higher_than_dissimilar() -> None:
    embedder = HashEmbedder(dimension=512)
    base = embedder.embed(["solar power energy sun"])[0]
    similar = embedder.embed(["solar energy from the sun"])[0]
    dissimilar = embedder.embed(["banana bread recipe flour"])[0]
    assert float(base @ similar) > float(base @ dissimilar)


def test_embedder_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimension=0)


def test_generator_uses_context() -> None:
    generator = TemplateGenerator(max_passages=2)
    answer = generator.generate("what is x?", ["context one", "context two", "context three"])
    assert "context one" in answer
    assert "context two" in answer
    # respects max_passages
    assert "context three" not in answer


def test_generator_handles_empty_context() -> None:
    generator = TemplateGenerator()
    answer = generator.generate("unanswerable?", [])
    assert "No relevant context" in answer
