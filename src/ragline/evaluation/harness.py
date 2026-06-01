"""The offline evaluation harness: run the pipeline over labeled QA pairs and
aggregate retrieval and faithfulness metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragline.evaluation.metrics import (
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from ragline.pipeline import Pipeline


@dataclass(frozen=True)
class EvalExample:
    """A labeled evaluation example."""

    question: str
    relevant_chunk_ids: set[str]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics across an evaluation dataset."""

    examples: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_faithfulness: float
    k: int

    def format_text(self) -> str:
        return (
            f"Evaluation over {self.examples} example(s) (k={self.k}):\n"
            f"  precision@{self.k}: {self.mean_precision_at_k:.3f}\n"
            f"  recall@{self.k}:    {self.mean_recall_at_k:.3f}\n"
            f"  MRR:           {self.mean_reciprocal_rank:.3f}\n"
            f"  faithfulness:  {self.mean_faithfulness:.3f}"
        )


def load_dataset(path: Path) -> list[EvalExample]:
    """Load evaluation examples from a JSON Lines file.

    Each line must be a JSON object with `question` (str) and
    `relevant_chunk_ids` (list of str).
    """
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            question = record.get("question")
            relevant = record.get("relevant_chunk_ids")
            if not isinstance(question, str) or not isinstance(relevant, list):
                raise ValueError(f"malformed example on line {line_no}")
            examples.append(
                EvalExample(question=question, relevant_chunk_ids=set(relevant))
            )
    return examples


def evaluate(pipeline: Pipeline, dataset: list[EvalExample], k: int = 5) -> EvaluationReport:
    """Run the pipeline over each example and aggregate metrics."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not dataset:
        return EvaluationReport(0, 0.0, 0.0, 0.0, 0.0, k)

    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    faithfulness_sum = 0.0

    for example in dataset:
        result = pipeline.query(example.question)
        retrieved_ids = [scored.chunk.chunk_id for scored in result.sources]
        context = [scored.chunk.text for scored in result.sources]

        precision_sum += precision_at_k(retrieved_ids, example.relevant_chunk_ids, k)
        recall_sum += recall_at_k(retrieved_ids, example.relevant_chunk_ids, k)
        mrr_sum += mean_reciprocal_rank(retrieved_ids, example.relevant_chunk_ids)
        faithfulness_sum += faithfulness(result.answer, context)

    n = len(dataset)
    return EvaluationReport(
        examples=n,
        mean_precision_at_k=precision_sum / n,
        mean_recall_at_k=recall_sum / n,
        mean_reciprocal_rank=mrr_sum / n,
        mean_faithfulness=faithfulness_sum / n,
        k=k,
    )
