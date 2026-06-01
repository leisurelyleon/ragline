#!/usr/bin/env bash
# Ingest the sample corpus, run a query, and run the evaluation harness.
# Fully offline: uses the deterministic local provider, no API key needed.
set -euo pipefail

echo "== Ingest corpus =="
ragline ingest data/corpus

echo
echo "== Query =="
ragline query "How do solar panels generate electricity?" --corpus data/corpus

echo
echo "== Evaluate =="
ragline eval data/eval/qa.jsonl --corpus data/corpus
