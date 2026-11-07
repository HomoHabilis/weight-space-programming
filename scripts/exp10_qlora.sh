#!/usr/bin/env bash
# Experiment 10, QLoRA baseline: fine-tune 4B on its own passing practice answers, then score it on the scorecard.
# Usage: scripts/exp10_qlora.sh [name=qlora-4b] [wait for pid ...]
# 1. installs the fast linear-attention kernels transformers uses for Qwen3.5 (flash-linear-attention; causal-conv1d
#    if it builds). Without them PyTorch's reference version runs out of GPU memory on one 10k-token example.
#    They change speed and memory, not results. vLLM (all scoring) has its own kernels and is not affected.
#    Installed with `uv pip`, so a later plain `uv run` may remove them again; this script reinstalls them.
# 2. probes one training step on the longest example at max lengths 14000, 12000, 10000, 8000 tokens (each in a fresh
#    process: a CUDA out-of-memory error cannot be recovered in-process) and keeps the first that fits
# 3. trains and evaluates (python -m wsp.exp10.lora) with that max length; longer examples are skipped and counted
# Progress: runs/exp10/<name>/train.log
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-qlora-4b}; shift $(( $# < 1 ? $# : 1 ))
OUT="runs/exp10/$NAME"; LOG="$OUT/train.log"
mkdir -p "$OUT"
# only one copy at a time (remove the lock folder to run again)
mkdir "$OUT/.lock" 2>/dev/null || { echo "$OUT/.lock exists: another run of this script started already"; exit 0; }
[ -f "$LOG" ] && mv "$LOG" "$OUT/train-attempt-$(date +%Y%m%d-%H%M%S).log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
PY=.venv/bin/python   # not `uv run`: it would sync the environment and remove the kernels installed below

for pid in "$@"; do
  say "waiting for process $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

say "installing fast kernels"
uv pip install --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
timeout 1800 uv pip install --python "$PY" causal-conv1d 2>&1 | tail -3 | tee -a "$LOG" \
  || say "causal-conv1d did not install (optional: its fallback is a plain convolution)"

FIT=""
for L in 14000 12000 10000 8000; do
  say "probe: max length $L"
  rc=0; "$PY" -m wsp.exp10.lora --name "$NAME" --probe --max-len "$L" >> "$LOG" 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then FIT=$L; break; fi
  [ "$rc" = 3 ] || { say "probe failed (exit $rc), not out of memory: see $LOG"; exit 1; }
done
[ -n "$FIT" ] || { say "no max length fits in GPU memory; see $OUT/probe.json"; exit 1; }
say "training with max length $FIT"
"$PY" -m wsp.exp10.lora --name "$NAME" --max-len "$FIT" >> "$LOG" 2>&1
say "done: $OUT/summary.json"
