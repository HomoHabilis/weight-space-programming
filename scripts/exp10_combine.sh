#!/usr/bin/env bash
# Experiment 10: combine the two Opus edits (thinking run + no-thinking run) if each beats QLoRA in its own mode,
# then score the combination on the scorecard in both modes.
# Usage: scripts/exp10_combine.sh [edit 1=code-opus-4b] [edit 2=nothink-opus-4b] [qlora=qlora-4b]
#   edit 1 is compared with QLoRA with thinking on (8k budget), edit 2 with thinking off (12k): held-out pass@1.
#   If QLoRA could not be trained (out of GPU memory: summary.json "fits_in_gpu": false) the edits count as better.
#   The two edits must change different tensors (they are put together as they are, nothing is averaged).
# Decision and scores: runs/exp10/<edit 1>+<edit 2>.decision.json; scores runs/exp10_eval/combined-4b-*.json
set -euo pipefail
cd "$(dirname "$0")/.."
E1=${1:-code-opus-4b}; E2=${2:-nothink-opus-4b}; Q=${3:-qlora-4b}
NAME=combined-4b
PY=${COMBINE_PY:-.venv/bin/python}
OUT="runs/exp10/$NAME/edit"
LOG="runs/exp10/$NAME.log"
mkdir -p "runs/exp10/$NAME"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

say "comparing $E1 and $E2 with $Q"
VERDICT=$("$PY" - "$E1" "$E2" "$Q" "$NAME" <<'EOF'
import json, sys
from pathlib import Path
e1, e2, q, name = sys.argv[1:]
R = Path("runs/exp10_eval")
def score(n):
    f = R / f"{n}.json"
    return json.loads(f.read_text())["pass@1"] if f.exists() else None
rec = {"edit1": e1, "edit2": e2, "qlora": q,
       "edit1_think8k": score(f"{e1}-think8k"), "edit2_nothink12k": score(f"{e2}-nothink12k"),
       "qlora_think8k": score(f"{q}-think8k"), "qlora_nothink12k": score(f"{q}-nothink12k")}
summ = Path(f"runs/exp10/{q}/summary.json")
s = json.loads(summ.read_text()) if summ.exists() else {}
pf = Path(f"runs/exp10/{q}/probe.json")
probe = json.loads(pf.read_text()) if pf.exists() else {}
if rec["edit1_think8k"] is None or rec["edit2_nothink12k"] is None:
    rec["verdict"], why = "STOP", "an edit has no held-out score"
elif s.get("fits_in_gpu") is False or (not s and probe.get("fits_in_gpu") is False):
    rec["verdict"], why = "COMBINE", "QLoRA could not be trained on this GPU"
elif rec["qlora_think8k"] is None or rec["qlora_nothink12k"] is None:
    rec["verdict"], why = "STOP", "QLoRA has no score in both modes (did it finish?)"
else:
    better = rec["edit1_think8k"] > rec["qlora_think8k"] and rec["edit2_nothink12k"] > rec["qlora_nothink12k"]
    rec["verdict"] = "COMBINE" if better else "NO-COMBINE"
    why = (f"edit 1 {rec['edit1_think8k']:.3f} vs QLoRA {rec['qlora_think8k']:.3f} (thinking), "
           f"edit 2 {rec['edit2_nothink12k']:.3f} vs QLoRA {rec['qlora_nothink12k']:.3f} (no thinking)")
rec["why"] = why
Path(f"runs/exp10/{e1}+{e2}.decision.json").write_text(json.dumps(rec, indent=1))
print(rec["verdict"])
print(f"{rec['verdict']}: {why}", file=sys.stderr)
EOF
) 2> >(tee -a "$LOG" >&2)
say "verdict: $VERDICT"
[ "$VERDICT" = COMBINE ] || exit 0
[ -n "${COMBINE_PLAN_ONLY:-}" ] && { echo "plan only: would combine"; exit 0; }

rm -rf "$OUT"; mkdir -p "$OUT"
for f in runs/agent/"$E1"/edit/*.npy runs/agent/"$E2"/edit/*.npy; do
  b=$(basename "$f")
  [ -e "$OUT/$b" ] && { say "both edits change $b: not combined"; exit 1; }
  cp "$f" "$OUT/$b"
done
say "combined edit: $(ls "$OUT" | tr '\n' ' ')"
EVAL=("$PY" -m wsp.exp10.eval --model base4b --edit-dir "$OUT" --set test --difficulty easy,medium --kv-fp8
      --max-num-seqs 12)
"${EVAL[@]}" --thinking --think-budget 8192 --answer-tokens 4096 --name "$NAME-think8k" >> "$LOG" 2>&1
"${EVAL[@]}" --max-tokens 12288 --name "$NAME-nothink12k" >> "$LOG" 2>&1
say "done: runs/exp10_eval/$NAME-think8k.json, runs/exp10_eval/$NAME-nothink12k.json"
