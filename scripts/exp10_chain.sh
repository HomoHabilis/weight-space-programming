#!/usr/bin/env bash
# Experiment 10, second editor run (thinking off) stacked on the first run's edit (thinking on), if that edit helped.
# Usage: scripts/exp10_chain.sh <first run> [min gain=0.05] [wait for pid ...]
#   scripts/exp10_chain.sh code-opus-4b 0.05 <pid of the QLoRA job>
# Steps (each stops the chain if it fails):
#   1. wait for the given processes (GPU jobs before this one) to end
#   2. go/no-go: the first edit's held-out score (thinking, 8k budget) must beat the base model's by >= min gain
#   3. write the stacked starting model: base 4B with the first edit in it (models/stacked/<first run>/Qwen3.5-4B)
#   4. GPU dry run of the no-thinking harness on it (~15 min)
#   5. the starting model's practice samples, thinking off (~1 h)
#   6. the second editor run (Opus, 24 passes), then its edit scored on the scorecard in both modes
# Progress: runs/exp10/chain-<first run>.log
set -euo pipefail
cd "$(dirname "$0")/.."
FIRST=${1:?first run name}; GAIN=${2:-0.05}; shift $(( $# < 2 ? $# : 2 ))
SECOND="$FIRST-nothink"
LOG="runs/exp10/chain-$FIRST.log"
mkdir -p runs/exp10
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
PY=(uv run python)

for pid in "$@"; do
  say "waiting for process $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

say "step 2: go/no-go"
"${PY[@]}" - "$FIRST" "$GAIN" <<'EOF' | tee -a "$LOG"
import json, sys
from pathlib import Path
first, gain = sys.argv[1], float(sys.argv[2])
R = Path("runs/exp10_eval")
base = json.loads((R / "base4b-think8k-em.json").read_text())
f = R / f"{first}-think8k.json"
if not f.exists() or not any(Path(f"runs/agent/{first}/edit").glob("*.npy")):
    print(f"NO-GO: no scored edit for {first}"); sys.exit(1)
edit = json.loads(f.read_text())
assert edit["problems"] == base["problems"] and edit["n"] == base["n"], "different problem sets"
d = edit["pass@1"] - base["pass@1"]
verdict = "GO" if d >= gain else "NO-GO"
Path(f"runs/exp10/chain-{first}.decision.json").write_text(json.dumps(
    {"base": base["pass@1"], "edit": edit["pass@1"], "gain": round(d, 4), "min_gain": gain, "verdict": verdict}, indent=1))
print(f"{verdict}: base {base['pass@1']:.3f}, edit {edit['pass@1']:.3f}, gain {d:+.3f} (need {gain:+.3f})")
sys.exit(0 if verdict == "GO" else 1)
EOF

STACK="models/stacked/$FIRST/Qwen3.5-4B"   # folder name matches the 4B profile (4-bit, prompt description)
say "step 3: stacked model $STACK"
"${PY[@]}" -c "
from pathlib import Path
from wsp.exp8.common import resolve_model
from wsp.exp10.generate import edited_dir, prior_edits
out = edited_dir(resolve_model('base4b'), Path('runs/agent/$FIRST/edit'), Path('$STACK'))
print(out, 'edited tensors:', prior_edits(out))" | tee -a "$LOG"
STACK=$(realpath "$STACK")

say "step 4: dry run (no thinking, stacked)"
"${PY[@]}" tests/agent_dry_run_code.py --no-thinking --target "$STACK" > "runs/exp10/chain-$FIRST.dryrun.log" 2>&1 \
  || { say "dry run FAILED: see runs/exp10/chain-$FIRST.dryrun.log"; exit 1; }
tail -6 "runs/exp10/chain-$FIRST.dryrun.log" | tee -a "$LOG"
rm -rf runs/agent/dry-run-code-nothink

SAMPLES="samples-$FIRST-nothink12k-n3"
say "step 5: practice samples $SAMPLES"
"${PY[@]}" -m wsp.exp10.eval --model "$STACK" --set practice --difficulty easy,medium --max-tokens 12288 --n 3 \
  --max-num-seqs 12 --kv-fp8 --name "$SAMPLES" > "runs/exp10/chain-$FIRST.samples.log" 2>&1
BASE_SCORE=$("${PY[@]}" -c "
import json; d = json.load(open('runs/exp10_eval/$FIRST-nothink12k.json'))
print(f\"about {round(100 * d['pass@1'])}%\")")
say "samples done; starting model's held-out score with thinking off: $BASE_SCORE"

say "step 6: editor run $SECOND"
mkdir -p "runs/agent/$SECOND"
if ! pgrep -f watchdog_claude_calls.sh > /dev/null; then
  nohup bash scripts/watchdog_claude_calls.sh 1200 >> "runs/agent/$SECOND/watchdog.log" 2>&1 &
fi
if [ -f "runs/agent/$SECOND/state.json" ]; then
  "${PY[@]}" -m wsp.exp8.agent --resume "runs/agent/$SECOND" | tee -a "runs/agent/$SECOND/console.log"
else
  "${PY[@]}" -m wsp.exp8.agent --task code --no-thinking --target "$STACK" --samples "runs/exp10_eval/$SAMPLES" \
    --base-score "$BASE_SCORE" --backend claude-code --model claude-opus-5-5 --effort xhigh --passes 24 \
    --name "$SECOND" | tee "runs/agent/$SECOND/console.log"
fi
EVAL=("${PY[@]}" -m wsp.exp10.eval --model "$STACK" --edit-dir "runs/agent/$SECOND/edit" --set test
      --difficulty easy,medium --kv-fp8 --max-num-seqs 12)
if ! compgen -G "runs/agent/$SECOND/edit/*.npy" > /dev/null; then say "no edit submitted by $SECOND"; exit 1; fi
say "scoring $SECOND"
"${EVAL[@]}" --max-tokens 12288 --name "$SECOND-nothink12k"
"${EVAL[@]}" --thinking --think-budget 8192 --answer-tokens 4096 --name "$SECOND-think8k"
say "chain done"
