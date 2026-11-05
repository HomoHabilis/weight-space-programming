#!/usr/bin/env bash
# Experiment 9 baselines on Qwen3.5-0.8B: controls, closed-form edit (no gradients), LoRA. About 30 min on a 4070 laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m wsp.exp9.make_data
uv run python -m wsp.exp9.eval --model base                 # must run first: records the retention reference
uv run python -m wsp.exp9.eval --model base --in-context
uv run python -m wsp.exp9.rank_one --layers 18-22 --alpha 1 --lam 100 --protect-weight 8 --save r1-L18-22
uv run python -m wsp.exp9.eval --model models/edited/r1-L18-22
uv run python -m wsp.exp9.lora --name lora-replay-e2 --replay --epochs 2
uv run python -m wsp.collect_results
