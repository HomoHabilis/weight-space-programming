#!/usr/bin/env bash
# Experiment 10 with an LLM editor: make Qwen3.5 better at LiveCodeBench (thinking budget 8k) by editing weights.
# Usage: scripts/exp10_agent.sh <run name> [backend] [model] [effort] [passes] [target]
#   scripts/exp10_agent.sh code-opus-4b claude-code claude-opus-5-5 xhigh 24 base4b
# Re-running with the same name resumes a stopped run. Follow it with: tail -f runs/agent/<name>/console.log
# Afterwards the edit is scored on the held-out scorecard (95 easy+medium v6 problems), both modes.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?run name}; BACKEND=${2:-claude-code}; MODEL=${3:-}; EFFORT=${4:-xhigh}; PASSES=${5:-24}; TARGET=${6:-base4b}
mkdir -p "runs/agent/$NAME"
if [ -f "runs/agent/$NAME/state.json" ]; then
  uv run python -m wsp.exp8.agent --resume "runs/agent/$NAME" | tee -a "runs/agent/$NAME/console.log"
else
  M=(); [ -n "$MODEL" ] && M=(--model "$MODEL")
  uv run python -m wsp.exp8.agent --task code --backend "$BACKEND" "${M[@]}" --effort "$EFFORT" --passes "$PASSES" \
    --target "$TARGET" --name "$NAME" | tee "runs/agent/$NAME/console.log"
fi
EVAL=(uv run python -m wsp.exp10.eval --model "$TARGET" --edit-dir "runs/agent/$NAME/edit" --set test
      --difficulty easy,medium --kv-fp8 --max-num-seqs 12)
"${EVAL[@]}" --thinking --think-budget 8192 --answer-tokens 4096 --name "$NAME-think8k"
"${EVAL[@]}" --max-tokens 12288 --name "$NAME-nothink12k"
