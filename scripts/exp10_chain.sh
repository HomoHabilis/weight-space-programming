#!/usr/bin/env bash
# Experiment 10, second editor run with thinking off, after a first run with thinking on.
# Usage: scripts/exp10_chain.sh <first run> [min gain=0.05] [wait for pid ...]
#   scripts/exp10_chain.sh code-opus-4b 0.05 <pid of the QLoRA job>
# Steps (each stops the chain if it fails):
#   1. wait for the given processes (GPU jobs before this one) to end
#   2. go/no-go: does the first edit's held-out score (thinking, 8k budget) beat the base model's by >= min gain?
#        GO:    the second run starts from the first edit (stacked model: models/stacked/<first run>/Qwen3.5-4B),
#               run name <first run>-nothink
#        NO-GO: the second run starts from the unedited base model instead, run name nothink-opus-4b
#   3. GPU dry run of the no-thinking harness on the starting model (~15 min)
#   4. the starting model's practice samples, thinking off (~1 h)
#   5. the second editor run (Opus, 24 passes), then its edit scored on the scorecard in both modes
# Progress: runs/exp10/chain-<first run>.log
set -euo pipefail
cd "$(dirname "$0")/.."
FIRST=${1:?first run name}; GAIN=${2:-0.05}; shift $(( $# < 2 ? $# : 2 ))
LOG="runs/exp10/chain-$FIRST.log"
mkdir -p runs/exp10
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
PY=(${CHAIN_PY:-uv run python})  # CHAIN_PY: another interpreter (tests)

for pid in "$@"; do
  say "waiting for process $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
done

say "step 2: go/no-go"
VERDICT=$("${PY[@]}" - "$FIRST" "$GAIN" <<'EOF'
import json, sys
from pathlib import Path
first, gain = sys.argv[1], float(sys.argv[2])
R = Path("runs/exp10_eval")
base = json.loads((R / "base4b-think8k-em.json").read_text())
f = R / f"{first}-think8k.json"
if not f.exists() or not any(Path(f"runs/agent/{first}/edit").glob("*.npy")):
    rec = {"base": base["pass@1"], "edit": None, "gain": None, "min_gain": gain, "verdict": "NO-GO"}
    msg = f"NO-GO: no scored edit for {first}"
else:
    edit = json.loads(f.read_text())
    assert edit["problems"] == base["problems"] and edit["n"] == base["n"], "different problem sets"
    d = edit["pass@1"] - base["pass@1"]
    rec = {"base": base["pass@1"], "edit": edit["pass@1"], "gain": round(d, 4), "min_gain": gain,
           "verdict": "GO" if d >= gain else "NO-GO"}
    msg = f"{rec['verdict']}: base {base['pass@1']:.3f}, edit {edit['pass@1']:.3f}, gain {d:+.3f} (need {gain:+.3f})"
Path(f"runs/exp10/chain-{first}.decision.json").write_text(json.dumps(rec, indent=1))
print(rec["verdict"], file=sys.stdout)
print(msg, file=sys.stderr)
EOF
) 2> >(tee -a "$LOG" >&2)
say "verdict: $VERDICT"
case "$VERDICT" in GO|NO-GO) ;; *) say "go/no-go check failed"; exit 1;; esac

if [ "$VERDICT" = "GO" ]; then
  TARGET="models/stacked/$FIRST/Qwen3.5-4B"   # folder name matches the 4B profile (4-bit, prompt description)
  say "step 2b: stacked model $TARGET"
  "${PY[@]}" -c "
from pathlib import Path
from wsp.exp8.common import resolve_model
from wsp.exp10.generate import edited_dir, prior_edits
out = edited_dir(resolve_model('base4b'), Path('runs/agent/$FIRST/edit'), Path('$TARGET'))
print(out, 'edited tensors:', prior_edits(out))" | tee -a "$LOG"
  TARGET=$(realpath "$TARGET")
  SECOND="$FIRST-nothink"
  SAMPLES="samples-$FIRST-nothink12k-n3"
  START_SCORE="runs/exp10_eval/$FIRST-nothink12k.json"
else
  TARGET=base4b
  SECOND="nothink-opus-4b"
  SAMPLES="samples-4b-nothink12k-n3"
  START_SCORE="runs/exp10_eval/base4b-nothink12k-em.json"
fi
say "second run $SECOND, starting from $TARGET"
[ -n "${CHAIN_PLAN_ONLY:-}" ] && { echo "plan only: $SECOND $TARGET $SAMPLES $START_SCORE"; exit 0; }

say "step 3: dry run (no thinking)"
"${PY[@]}" tests/agent_dry_run_code.py --no-thinking --target "$TARGET" > "runs/exp10/chain-$FIRST.dryrun.log" 2>&1 \
  || { say "dry run FAILED: see runs/exp10/chain-$FIRST.dryrun.log"; exit 1; }
tail -6 "runs/exp10/chain-$FIRST.dryrun.log" | tee -a "$LOG"
rm -rf runs/agent/dry-run-code-nothink

say "step 4: practice samples $SAMPLES"
if [ ! -f "runs/exp10_eval/$SAMPLES.json" ]; then
  "${PY[@]}" -m wsp.exp10.eval --model "$TARGET" --set practice --difficulty easy,medium --max-tokens 12288 --n 3 \
    --max-num-seqs 12 --kv-fp8 --name "$SAMPLES" > "runs/exp10/chain-$FIRST.samples.log" 2>&1
fi
BASE_SCORE=$("${PY[@]}" -c "
import json; d = json.load(open('$START_SCORE'))
print(f\"about {round(100 * d['pass@1'])}%\")")
say "samples done; starting model's held-out score with thinking off: $BASE_SCORE"

say "step 5: editor run $SECOND"
mkdir -p "runs/agent/$SECOND"
if ! pgrep -f watchdog_claude_calls.sh > /dev/null; then
  nohup bash scripts/watchdog_claude_calls.sh 1200 >> "runs/agent/$SECOND/watchdog.log" 2>&1 &
fi
if [ -f "runs/agent/$SECOND/state.json" ]; then
  "${PY[@]}" -m wsp.exp8.agent --resume "runs/agent/$SECOND" | tee -a "runs/agent/$SECOND/console.log"
else
  "${PY[@]}" -m wsp.exp8.agent --task code --no-thinking --target "$TARGET" --samples "runs/exp10_eval/$SAMPLES" \
    --base-score "$BASE_SCORE" --backend claude-code --model claude-opus-5-5 --effort xhigh --passes 24 \
    --name "$SECOND" | tee "runs/agent/$SECOND/console.log"
fi
if ! compgen -G "runs/agent/$SECOND/edit/*.npy" > /dev/null; then say "no edit submitted by $SECOND"; exit 1; fi
EVAL=("${PY[@]}" -m wsp.exp10.eval --model "$TARGET" --edit-dir "runs/agent/$SECOND/edit" --set test
      --difficulty easy,medium --kv-fp8 --max-num-seqs 12)
say "scoring $SECOND"
"${EVAL[@]}" --max-tokens 12288 --name "$SECOND-nothink12k"
"${EVAL[@]}" --thinking --think-budget 8192 --answer-tokens 4096 --name "$SECOND-think8k"
say "chain done"
