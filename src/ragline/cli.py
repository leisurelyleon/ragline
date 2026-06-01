"""Command-line interface for ragline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ragline.document import Document
from ragline.evaluation.harness import evaluate, load_dataset
from ragline.pipeline import Pipeline
from ragline.providers.base import Embedder, Generator
from ragline.providers.local import HashEmbedder, TemplateGenerator

_DEFAULT_CORPUS = "data/corpus"


def _build_providers(provider: str) -> tuple[Embedder, Generator]:
    """Construct the embedder/generator pair for the named provider."""
    if provider == "local":
        return HashEmbedder(), TemplateGenerator()
    if provider == "openai":
        # Imported lazily so the optional dependency is only needed on demand.
        from ragline.providers.openai import OpenAIEmbedder, OpenAIGenerator

        return OpenAIEmbedder(), OpenAIGenerator()
    raise ValueError(f"unknown provider: {provider}")


def _load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every .md and .txt file in a directory as a document."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    documents: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append(Document(doc_id=path.name, text=path.read_text(encoding="utf-8")))
    return documents


def _build_ingested_pipeline(provider: str, corpus_dir: Path, top_k: int) -> Pipeline:
    """Build a pipeline and ingest a corpus into it, in one process."""
    embedder, generator = _build_providers(provider)
    pipeline = Pipeline(embedder, generator, top_k=top_k)
    pipeline.ingest(_load_corpus(corpus_dir))
    return pipeline


def _cmd_ingest(args: argparse.Namespace) -> int:
    documents = _load_corpus(Path(args.corpus))
    embedder, generator = _build_providers(args.provider)
    pipeline = Pipeline(embedder, generator)
    added = pipeline.ingest(documents)
    print(f"Loaded {len(documents)} document(s); indexed {added} chunk(s).")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = _build_ingested_pipeline(args.provider, Path(args.corpus), args.top_k)
    result = pipeline.query(args.question)
    print(f"Q: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Sources:")
    for scored in result.sources:
        print(f"  [{scored.score:.3f}] {scored.chunk.chunk_id}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    pipeline = _build_ingested_pipeline(args.provider, Path(args.corpus), args.k)
    dataset = load_dataset(Path(args.dataset))
    report = evaluate(pipeline, dataset, args.k)
    print(report.format_text())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragline",
        description="Retrieval-augmented generation with offline evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared --provider option, available on every subcommand in any position.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--provider",
        default="local",
        choices=["local", "openai"],
        help="embedding/generation provider (default: local, no API key needed)",
    )

    ingest = subparsers.add_parser("ingest", parents=[common], help="load and index a corpus")
    ingest.add_argument("corpus", nargs="?", default=_DEFAULT_CORPUS)
    ingest.set_defaults(func=_cmd_ingest)

    query = subparsers.add_parser("query", parents=[common], help="answer a question")
    query.add_argument("question")
    query.add_argument("--corpus", default=_DEFAULT_CORPUS)
    query.add_argument("--top-k", type=int, default=5, dest="top_k")
    query.set_defaults(func=_cmd_query)

    ev = subparsers.add_parser("eval", parents=[common], help="evaluate on a dataset")
    ev.add_argument("dataset")
    ev.add_argument("--corpus", default=_DEFAULT_CORPUS)
    ev.add_argument("--k", type=int, default=5)
    ev.set_defaults(func=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result)
