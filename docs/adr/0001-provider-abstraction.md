# 1. Provider abstraction for embeddings and generation

- Status: Accepted
- Date: 2026-05

## Context

RAG systems depend on embedding and language models, which are typically remote,
paid, non-deterministic API services. Coupling the pipeline directly to such a
service makes it impossible to test or reproduce reliably.

## Decision

Define `Embedder` and `Generator` as protocols. The pipeline depends only on
these interfaces. Provide deterministic local implementations as the default,
and an optional remote provider behind an extra dependency.

## Consequences

- The full pipeline is testable and reproducible with no network or API key.
- Swapping in a real provider requires no pipeline changes.
- The optional provider's dependency is never pulled by the default install.
