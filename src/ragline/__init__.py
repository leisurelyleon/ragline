"""ragline: a retrieval-augmented generation pipeline with offline evaluation."""

from __future__ import annotations

from ragline.document import Chunk, Document
from ragline.pipeline import Pipeline, QueryResult

__all__ = ["Chunk", "Document", "Pipeline", "QueryResult"]
__version__ = "0.1.0"
