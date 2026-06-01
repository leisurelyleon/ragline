#!/usr/bin/env bash
# Create a virtual environment and install ragline with dev tools.
set -euo pipefail

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
echo "Setup complete. Activate with: source .venv/bin/activate"
