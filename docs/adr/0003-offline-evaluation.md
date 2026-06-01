# 3. Offline, first-class evaluation

- Status: Accepted
- Date: 2026-05

## Context

Assembling a RAG pipeline is easy; knowing whether it works is hard. Quality
that is not measured cannot be trusted or improved.

## Decision

Make evaluation a first-class part of the system: a harness that runs the
pipeline over a labeled dataset and reports retrieval metrics (precision@k,
recall@k, MRR) and a faithfulness proxy, deterministically and offline.

## Consequences

- Retrieval and grounding quality are measured, not assumed.
- Regressions are detectable by re-running the harness.
- The faithfulness metric is a documented lexical heuristic, not a model judge.
