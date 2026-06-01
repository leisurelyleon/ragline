"""Offline evaluation: retrieval and faithfulness metrics, and the harness."""

from __future__ import annotations

from ragline.evaluation.harness import EvaluationReport, EvalExample, evaluate, load_dataset
from ragline.evaluation.metrics import (
    faithfulness,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "EvalExample",
    "EvaluationReport",
    "evaluate",
    "faithfulness",
    "load_dataset",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
]
