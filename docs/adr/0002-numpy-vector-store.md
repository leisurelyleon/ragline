# 2. In-memory NumPy vector store

- Status: Accepted
- Date: 2026-05

## Context

Retrieval needs vector similarity search. A dedicated vector database would add
an external service dependency, complicating setup, testing, and CI.

## Decision

Implement an in-memory vector store backed by a NumPy matrix, using cosine
similarity (a dot product over L2-normalized vectors) and `argsort` for top-k.

## Consequences

- Real vector-similarity math with no external service to run.
- Suitable for the corpus sizes a portfolio/demo project handles.
- A production deployment could swap in a dedicated store behind the same
  retrieval interface.
