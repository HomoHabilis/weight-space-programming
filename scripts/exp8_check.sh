#!/usr/bin/env bash
# Experiment 8 step 0: base Qwen3.5-0.8B vs the authors' published edit, on your GPU.
# Usage: scripts/exp8_check.sh [problems per set] [max tokens]     (default 50 and 32768: about 1 h on a 4070 laptop)
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-50}; T=${2:-32768}
uv run python -m wsp.exp8.eval_lite --models base published-edit --sets "math500:$N" "gsm8k:$N" --max-tokens "$T" \
  --out "check-$N-$T"
uv run python -m wsp.collect_results
