#!/usr/bin/env bash
# One-time setup: Python environment (vLLM, PyTorch, ...), the authors' logs, Experiment 9 data, free checks.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra gpu --extra kernel --python 3.12
bash scripts/fetch_logs.sh
uv run python -m wsp.exp9.make_data
uv run python tests/smoke.py
echo
echo "Next: log in to a model backend (see README), e.g. 'claude' then /login for the Claude Code backend."
