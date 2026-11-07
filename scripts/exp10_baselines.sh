#!/usr/bin/env bash
# Experiment 10 baselines on Qwen3.5-4B (4-bit): the scorecard in both modes, the practice samples (the editor's
# material and the QLoRA training data). About 3 hours.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m wsp.exp10.lcb prepare
E=(uv run python -m wsp.exp10.eval --model base4b --kv-fp8 --max-num-seqs 12 --difficulty easy,medium)
"${E[@]}" --set test --max-tokens 12288 --name base4b-nothink12k-em
"${E[@]}" --set test --thinking --think-budget 8192 --answer-tokens 4096 --name base4b-think8k-em
"${E[@]}" --set practice --thinking --think-budget 8192 --answer-tokens 4096 --n 3 --name samples-4b-think8k-n3
uv run python -m wsp.exp10.editor_data
# QLoRA baseline: scripts/exp10_qlora.sh (about 28 h on an 8 GB laptop GPU; see docs/exp10_competitive_programming.md)
