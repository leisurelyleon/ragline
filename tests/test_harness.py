"""Tests for the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragline.document import Document
from ragline.evaluation.harness import EvalExample, evaluate, load_dataset
from ragline.pipeline import Pipeline
from ragline.providers.local import HashEmbedder, TemplateGenerator


def _ingested_pipeline() -> Pipeline:
    pipeline = Pipeline(HashEmbedder(dimension=256), TemplateGenerator(), chunk_size=200,
                        overlap=40, top_k=3)
    pipeline.ingest([
        Document(doc_id="solar.md", text="Solar panels convert sunlight into electricity."),
        Document(doc_id="wind.md", text="Wind turbines convert moving air into electricity."),
    ])
    return pipeline


def test_evaluate_empty_dataset() -> None:
    report = evaluate(_ingested_pipeline(), [], k=3)
    assert report.examples == 0
    assert report.mean_precision_at_k == 0.0


def test_evaluate_produces_metrics() -> None:
    pipeline = _ingested_pipeline()
    dataset = [
        EvalExample(question="How do solar panels work?", relevant_chunk_ids={"solar.md:0"}),
        EvalExample(question="What do wind turbines do?", relevant_chunk_ids={"wind.md:0"}),
    ]
    report = evaluate(pipeline, dataset, k=3)
    assert report.examples == 2
    assert 0.0 <= report.mean_precision_at_k <= 1.0
    assert 0.0 <= report.mean_recall_at_k <= 1.0
    assert 0.0 <= report.mean_reciprocal_rank <= 1.0
    assert 0.0 <= report.mean_faithfulness <= 1.0


def test_report_format_text() -> None:
    report = evaluate(_ingested_pipeline(), [], k=5)
    text = report.format_text()
    assert "precision@5" in text
    assert "MRR" in text


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    lines = [
        {"question": "q1", "relevant_chunk_ids": ["a:0", "b:1"]},
        {"question": "q2", "relevant_chunk_ids": ["c:0"]},
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    examples = load_dataset(path)
    assert len(examples) == 2
    assert examples[0].question == "q1"
    assert examples[0].relevant_chunk_ids == {"a:0", "b:1"}


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"question": "q", "relevant_chunk_ids": ["x:0"]}\n\n\n', encoding="utf-8"
    )
    assert len(load_dataset(path)) == 1


def test_load_dataset_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": 123}\n', encoding="utf-8")  # wrong types
    with pytest.raises(ValueError):
        load_dataset(path)


def test_load_dataset_rejects_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)
