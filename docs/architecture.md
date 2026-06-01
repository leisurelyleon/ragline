# Architecture

`ragline` is a Python package that implements retrieval-augmented generation
with a first-class offline evaluation harness. It is organized so the entire
pipeline runs deterministically with no network access or API key.

## Layout

```text
src/ragline/
document.py        Document and Chunk dataclasses
chunking.py        pure overlapping-chunk splitting
providers/         Embedder + Generator protocols; local (real) + openai (optional)
vector_store.py    NumPy cosine-similarity store with top-k retrieval
pipeline.py        chunk -> embed -> store -> retrieve -> generate
evaluation/        retrieval/faithfulness metrics + the evaluation harness
cli.py             ingest / query / eval
```

## The provider seam

`Embedder` and `Generator` are `Protocol`s. The pipeline depends only on these
interfaces, never on a concrete backend. `HashEmbedder` and `TemplateGenerator`
are deterministic, dependency-light implementations that make the whole system
reproducible offline. `OpenAIEmbedder`/`OpenAIGenerator` are optional, imported
only when explicitly selected.

## Determinism

Embeddings use SHA-256-based feature hashing rather than Python's randomized
`hash()`, so vectors are stable across runs and platforms. This is what makes
the evaluation harness reproducible: the same corpus and dataset always yield
the same metrics.

## Evaluation as a first-class concern

The `evaluation` package measures retrieval quality (precision@k, recall@k, MRR)
and a lexical faithfulness proxy, aggregated over a labeled dataset. See
[`evaluation.md`](evaluation.md).
