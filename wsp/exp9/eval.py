"""Score a model on the Experiment 9 knowledge base.

  python -m wsp.exp9.eval --model base                          # control: does it already know? (should be ~0)
  python -m wsp.exp9.eval --model base --in-context             # control: facts in the prompt (should be ~100)
  python -m wsp.exp9.eval --model models/edited/<run>           # an edited model
  python -m wsp.exp9.eval --model base --adapter runs/exp9/lora # LoRA baseline

Metrics (closed-book unless --in-context):
  efficacy        the 3 train phrasings per fact (the editor / LoRA may use these)
  generalization  the 4 held-out phrasings per fact (headline)
  bleed           neighbor questions (same question about another company) answered with this fact's answer (lower
                  is better)
  retention       real-world questions the BASE model gets right that are still right
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import ROOT, RUNS
from ..exp8.common import resolve_model
from .make_data import OUT as DATA
from .qa import Answerer, is_correct, messages


def load(name: str) -> list[dict]:
    p = DATA / name
    if not p.exists():
        raise SystemExit(f"{p} missing: run python -m wsp.exp9.make_data")
    return [json.loads(line) for line in open(p)]


def evaluate(model_dir: str, adapter: str | None = None, in_context: bool = False, answerer=None,
             set_reference: bool = False) -> dict:
    """set_reference: this IS the unedited base model, closed-book: record which retention questions it gets right."""
    facts, retention = load("facts.jsonl"), load("retention.jsonl")
    ctx = "\n".join(f["statement"] for f in facts) if in_context else None
    a = answerer or Answerer(model_dir, adapter)
    jobs = []  # (metric, fact or retention row, question, target)
    for f in facts:
        jobs += [("efficacy", f, q, f["answer"]) for q in f["train"]]
        jobs += [("generalization", f, q, f["answer"]) for q in f["test"]]
        jobs.append(("bleed", f, f["neighbor"], f["answer"]))
    jobs += [("retention", r, r["question"], r["answer"]) for r in retention]
    t0 = time.time()
    outs = a.answer([messages(q, ctx) for _, _, q, _ in jobs])
    rows = [{"metric": m, "id": x["id"], "kind": x.get("kind"), "question": q, "target": t, "answer": o,
             "correct": is_correct(o, t)} for (m, x, q, t), o in zip(jobs, outs)]

    base_ok_path = DATA / "retention_base.json"
    if set_reference:
        base_ok_path.write_text(json.dumps(sorted(r["id"] for r in rows if r["metric"] == "retention" and r["correct"])))
    base_ok = set(json.loads(base_ok_path.read_text())) if base_ok_path.exists() else None

    def rate(metric, kind=None):
        sel = [r for r in rows if r["metric"] == metric and (kind is None or r["kind"] == kind)]
        if metric == "retention" and base_ok is not None:
            sel = [r for r in sel if r["id"] in base_ok]
        return round(sum(r["correct"] for r in sel) / len(sel), 3) if sel else None

    kinds = sorted({f["kind"] for f in facts})
    res = {"model": str(model_dir), "adapter": adapter, "in_context": in_context, "s": round(time.time() - t0, 1),
           **{m: rate(m) for m in ("efficacy", "generalization", "bleed", "retention")},
           "retention_note": ("share of the base model's correct answers kept" if base_ok is not None
                              else "raw accuracy (run the base model first to set the reference)"),
           "generalization_by_kind": {k: rate("generalization", k) for k in kinds},
           "facts_learned": sum(1 for f in facts if sum(r["correct"] for r in rows
                                                        if r["id"] == f["id"] and r["metric"] == "generalization") >= 3),
           "rows": rows}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base")
    ap.add_argument("--adapter", help="a LoRA adapter folder (peft)")
    ap.add_argument("--in-context", action="store_true", help="put all fact statements in the system prompt")
    ap.add_argument("--name", help="results name under runs/exp9_eval/")
    ap.add_argument("--show", type=int, default=6, help="print this many example answers")
    args = ap.parse_args()
    model_dir = str(resolve_model(args.model))
    unedited = args.model == "base" and not args.adapter and not args.in_context
    res = evaluate(model_dir, args.adapter, args.in_context, set_reference=unedited)
    name = args.name or (Path(args.model).name + ("-lora" if args.adapter else "") + ("-context" if args.in_context
                                                                                        else ""))
    out = RUNS / "exp9_eval" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"{name}: efficacy {res['efficacy']}  generalization {res['generalization']}  bleed {res['bleed']}  "
          f"retention {res['retention']}  facts learned {res['facts_learned']}/30  ({res['s']} s)")
    print("  by kind:", res["generalization_by_kind"])
    for r in [r for r in res["rows"] if r["metric"] == "generalization"][:: max(1, 120 // max(args.show, 1))][:args.show]:
        print(f"  {'OK ' if r['correct'] else 'BAD'} {r['question'][:60]:<60} -> {r['answer'][:40]!r} (want {r['target']})")
    print(f"results: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
