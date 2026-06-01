"""End-to-end pipeline tests on the deterministic local providers."""

from __future__ import annotations

from ragline.document import Document
from ragline.pipeline import Pipeline
from ragline.providers.local import HashEmbedder, TemplateGenerator


def _pipeline() -> Pipeline:
    return Pipeline(
        HashEmbedder(dimension=256),
        TemplateGenerator(),
        chunk_size=80,
        overlap=20,
        top_k=3,
    )


def _corpus() -> list[Document]:
    return [
        Document(doc_id="solar.md", text="Solar panels convert sunlight into electricity "
                 "using photovoltaic cells made of silicon."),
        Document(doc_id="wind.md", text="Wind turbines convert the kinetic energy of moving "
                 "air into electricity using large rotating blades."),
        Document(doc_id="hydro.md", text="Hydroelectric dams convert the energy of falling "
                 "water into electricity by spinning turbines."),
    ]


def test_ingest_reports_chunk_count() -> None:
    pipeline = _pipeline()
    added = pipeline.ingest(_corpus())
    assert added > 0
    assert len(pipeline.store) == added


def test_ingest_empty_corpus_adds_nothing() -> None:
    pipeline = _pipeline()
    assert pipeline.ingest([]) == 0
    assert pipeline.store.is_empty()


def test_query_returns_grounded_answer() -> None:
    pipeline = _pipeline()
    pipeline.ingest(_corpus())
    result = pipeline.query("How do solar panels make electricity?")
    assert result.question.startswith("How do solar")
    assert result.answer  # non-empty
    assert len(result.sources) > 0
    # The most relevant source should be the solar document.
    assert result.sources[0].chunk.doc_id == "solar.md"


def test_query_on_empty_pipeline_is_graceful() -> None:
    pipeline = _pipeline()
    result = pipeline.query("anything?")
    assert result.sources == []
    assert "No relevant context" in result.answer
