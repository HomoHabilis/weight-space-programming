#!/usr/bin/env bash
# Experiment 9 with an LLM editor. Usage: scripts/exp9_agent.sh <run name> [backend] [model] [effort] [passes]
#   scripts/exp9_agent.sh facts-opus claude-code claude-opus-5-5 xhigh 30
#   scripts/exp9_agent.sh facts-astra copilot gpt-6-astra high 30
# Re-running with the same name resumes a stopped run.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?run name}; BACKEND=${2:-claude-code}; MODEL=${3:-}; EFFORT=${4:-xhigh}; PASSES=${5:-30}
mkdir -p "runs/exp8/$NAME"
if [ -f "runs/exp8/$NAME/state.json" ]; then
  uv run python -m wsp.exp8.agent --resume "runs/exp8/$NAME" | tee -a "runs/exp8/$NAME/console.log"
else
  M=(); [ -n "$MODEL" ] && M=(--model "$MODEL")
  uv run python -m wsp.exp8.agent --task facts --backend "$BACKEND" "${M[@]}" --effort "$EFFORT" --passes "$PASSES" \
    --name "$NAME" | tee "runs/exp8/$NAME/console.log"
fi
uv run python -m wsp.exp9.eval --model "models/edited/$NAME" --name "$NAME"
uv run python -m wsp.collect_results
