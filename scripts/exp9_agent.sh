#!/usr/bin/env bash
# Experiment 9 with an LLM editor.
# Usage: scripts/exp9_agent.sh <run name> [backend] [model] [effort] [passes] [target]
#   scripts/exp9_agent.sh facts-opus-v3 claude-code claude-opus-5-5 xhigh 30            # Qwen3.5-0.8B
#   scripts/exp9_agent.sh facts-opus-9b claude-code claude-opus-5-5 xhigh 30 base9b     # Qwen3.5-9B in 4-bit
#   scripts/exp9_agent.sh facts-astra-v3 copilot gpt-6-astra xhigh 30
# Re-running with the same name resumes a stopped run. Follow it with: tail -f runs/agent/<name>/console.log
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?run name}; BACKEND=${2:-claude-code}; MODEL=${3:-}; EFFORT=${4:-xhigh}; PASSES=${5:-30}; TARGET=${6:-base}
mkdir -p "runs/agent/$NAME"
if [ -f "runs/agent/$NAME/state.json" ]; then
  uv run python -m wsp.exp8.agent --resume "runs/agent/$NAME" | tee -a "runs/agent/$NAME/console.log"
else
  M=(); [ -n "$MODEL" ] && M=(--model "$MODEL")
  uv run python -m wsp.exp8.agent --task facts --backend "$BACKEND" "${M[@]}" --effort "$EFFORT" --passes "$PASSES" \
    --target "$TARGET" --name "$NAME" | tee "runs/agent/$NAME/console.log"
fi
TARGET=$(python3 -c "import json; print(json.load(open('runs/agent/$NAME/config.json')).get('target', 'base'))")
if [ "$TARGET" = "base" ]; then
  uv run python -m wsp.exp9.eval --model "models/edited/$NAME" --name "$NAME"
else
  uv run python -m wsp.exp9.eval --model "$TARGET" --edit-dir "runs/agent/$NAME/edit" --name "$NAME"
fi
uv run python -m wsp.collect_results
