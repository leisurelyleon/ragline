# ragline

> A retrieval-augmented generation pipeline with a rigorous offline evaluation harness.

`ragline` ingests documents, splits them into overlapping chunks, embeds them
into a vector store, retrieves the most relevant chunks for a question by cosine
similarity, and generates a grounded answer. Its defining feature is an offline
evaluation harness that measures retrieval quality and answer faithfulness
against a labeled dataset — so you can know whether the system actually works,
not just that it runs.

## The Problem

RAG systems are easy to assemble and hard to *trust*. Retrieval can return
irrelevant context; generation can ignore the context it was given. `ragline`
treats evaluation as a first-class concern, reporting precision@k, recall@k,
MRR, and a faithfulness check so quality is measured, not assumed.

## No API key required

Embedding and generation sit behind provider interfaces. The default providers
are deterministic and local — a hash-based embedder and a template generator —
so the entire pipeline, its tests, and the evaluation harness run with **no API
key and no network access**. A real provider (e.g. OpenAI) is available as an
optional dependency for production use.

## Architecture

```
src/ragline/
  document.py        documents and chunks
  chunking.py        split documents into overlapping chunks
  providers/         Embedder + Generator interfaces; local (real) + openai (optional)
  vector_store.py    NumPy cosine-similarity store with top-k retrieval
  pipeline.py        chunk -> embed -> store -> retrieve -> generate
  evaluation/        retrieval/faithfulness metrics + the eval harness
  cli.py             ingest / query / eval
```

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/evaluation.md`](docs/evaluation.md).

## Requirements

- Python 3.10+
- NumPy (installed automatically)

## Setup

```bash
pip install -e ".[dev]"
```

## Use

```bash
ragline ingest data/corpus
ragline query "How do solar panels generate electricity?"
ragline eval data/eval/qa.jsonl
```

## Develop

```bash
ruff check .
black --check .
mypy
pytest
```

## License

MIT — see [LICENSE](LICENSE).
