#!/usr/bin/env bash
# Experiment 9 baselines on Qwen3.5-4B in 4-bit (8 GB GPU). Downloads the model (8.8 GB) on first use. A few hours.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m wsp.exp9.make_data
uv run python -m wsp.exp9.eval --model base4b                 # must run first: records the 4B retention reference
uv run python -m wsp.exp9.eval --model base4b --in-context
uv run python -m wsp.exp9.rank_one --model base4b --layers 24-28 --alpha 1 --lam 100 --protect-weight 8 --save r1-4b-L24-28
uv run python -m wsp.exp9.eval --model base4b --edit-dir runs/exp9/r1-4b-L24-28/edit --name r1-4b-L24-28
# QLoRA: batch 8 first; if it runs out of memory (recorded as fits_in_gpu: false), batch 1 + gradient checkpointing
uv run python -m wsp.exp9.lora --model base4b --name qlora-4b-e2 --replay --epochs 2 --batch 8 || true
uv run python -m wsp.exp9.lora --model base4b --name qlora-4b-e2-b1 --replay --epochs 2 --batch 1 --grad-checkpointing
uv run python -m wsp.collect_results
