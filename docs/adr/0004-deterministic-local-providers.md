# 4. Deterministic local providers

- Status: Accepted
- Date: 2026-05

## Context

For tests and the evaluation harness to be reproducible, the default providers
must produce identical output across runs and machines.

## Decision

Implement embeddings via SHA-256-based feature hashing (not Python's
per-process randomized `hash()`), and generation via a deterministic template
that quotes retrieved context. Both are real, inspectable, and dependency-light.

## Consequences

- Identical results across runs and platforms.
- The evaluation harness yields stable, comparable metrics.
- The local providers are illustrative; the real provider is for production use.
