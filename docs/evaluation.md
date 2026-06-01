# Evaluation

`ragline`'s evaluation harness runs the pipeline over a labeled dataset and
reports four metrics. It is fully offline and deterministic.

## Dataset format

A JSON Lines file; each line is an object:

```json
{"question": "...", "relevant_chunk_ids": ["doc.md:0", "doc.md:1"]}
```

`relevant_chunk_ids` are the chunk identifiers a correct retrieval should
surface for that question.

## Metrics

- **precision@k** — of the top-k retrieved chunks, the fraction that are
  relevant. High precision means little irrelevant context.
- **recall@k** — of all relevant chunks, the fraction found within the top-k.
  High recall means the needed context was retrieved.
- **MRR (mean reciprocal rank)** — the reciprocal of the rank of the first
  relevant chunk, averaged over questions. Rewards putting a relevant chunk
  near the top.
- **faithfulness** — a lexical proxy: the fraction of the answer's content
  words that appear in the retrieved context. It is a heuristic for grounding,
  not a model-based judgment, and is documented as such.

## Reading the numbers

These metrics are diagnostic, not a single pass/fail. Low recall suggests the
chunking or embedding is missing relevant content; low precision suggests too
much irrelevant context is retrieved; low faithfulness suggests the generator
is drawing on knowledge outside the retrieved context.
