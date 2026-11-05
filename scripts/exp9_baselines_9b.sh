#!/usr/bin/env bash
# Experiment 9 baselines on Qwen3.5-9B in 4-bit (8 GB GPU). Downloads the model (19 GB) on first use. Several hours.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m wsp.exp9.make_data
uv run python -m wsp.exp9.eval --model base9b                 # must run first: records the 9B retention reference
uv run python -m wsp.exp9.eval --model base9b --in-context
uv run python -m wsp.exp9.rank_one --model base9b --layers 24-28 --alpha 1 --lam 100 --protect-weight 8 --save r1-9b-L24-28
uv run python -m wsp.exp9.eval --model base9b --edit-dir runs/exp9/r1-9b-L24-28/edit --name r1-9b-L24-28
# QLoRA: batch 8 runs out of memory; batch 1 + gradient checkpointing completes only by overflowing into system RAM
uv run python -m wsp.exp9.lora --model base9b --name qlora-9b-e2-b1 --replay --epochs 2 --batch 1 --grad-checkpointing
uv run python -m wsp.collect_results
