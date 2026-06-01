# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold: ragline package and CLI entry point.

## [0.1.0] - TBD

### Added
- Document chunking with configurable size and overlap.
- Pluggable embedding/generation providers with deterministic local fakes.
- NumPy-backed vector store with cosine-similarity top-k retrieval.
- End-to-end RAG pipeline: chunk, embed, store, retrieve, generate.
- Offline evaluation harness reporting precision@k, recall@k, MRR, and
  faithfulness against a labeled dataset.

[Unreleased]: https://github.com/leisurelyleon/ragline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/leisurelyleon/ragline/releases/tag/v0.1.0
