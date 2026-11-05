#!/usr/bin/env bash
# Experiments 1-2 (read a network's weights; no GPU). Usage: scripts/exp1_2.sh [backend] [model] [effort]
#   scripts/exp1_2.sh claude-code claude-opus-5-5 high
#   scripts/exp1_2.sh copilot gpt-6-astra high
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND=${1:-claude-code}; MODEL=${2:-}; EFFORT=${3:-high}
M=(); [ -n "$MODEL" ] && M=(--model "$MODEL")
uv run python -m wsp.read_weights score --exp 1 --published
uv run python -m wsp.read_weights run --exp 1 --backend "$BACKEND" "${M[@]}" --effort "$EFFORT"
uv run python -m wsp.read_weights run --exp 2 --backend "$BACKEND" "${M[@]}" --effort "$EFFORT"
uv run python -m wsp.collect_results
