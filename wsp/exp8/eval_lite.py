"""Evaluate Qwen3.5-0.8B checkpoints on a subset of MATH-500 / GSM8K with the paper's protocol, on one GPU.

Step 0 of the replication: confirm the authors' published edit (Astra, 50 passes) beats the base model on your GPU.

  python -m wsp.exp8.eval_lite --models base published-edit --sets math500:50 gsm8k:50 --max-tokens 32768
  python -m wsp.exp8.eval_lite --models base runs/exp8/<run>/edited_model --sets math500:100

Paper (full sets, 32,768 tokens): MATH-500 7.9 -> 48.1, GSM8K 13.8 -> 67.1, share cut at the limit 93 -> 33 %.
A smaller --max-tokens is much faster but changes the numbers (more answers get cut).
"""
from __future__ import annotations

import argparse
import json
import time

from .. import RUNS
from .common import eval_set, paired, resolve_model, summarize, user_message
from .generate import spawn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["base", "published-edit"],
                    help="'base', 'published-edit', a local folder or a HF repo id")
    ap.add_argument("--sets", nargs="+", default=["math500:50", "gsm8k:50"], help="name:count")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=32, help="concurrent sequences (lower if out of memory)")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = []
    for s in args.sets:
        name, _, cnt = s.partition(":")
        for r in eval_set(name, int(cnt) if cnt else None):
            items.append({"id": r["id"], "source": r["source"], "user": user_message(r["problem"]),
                          "answer": r["answer"]})
    out = RUNS / "exp8_eval" / (args.out or time.strftime("%Y%m%d-%H%M%S"))
    results, rows_by_model = {}, {}
    for spec in args.models:
        model_dir = resolve_model(spec)
        tag = spec.replace("/", "_")
        print(f"[eval] {spec} ({model_dir}) on {len(items)} problems x {args.samples}, max {args.max_tokens} tokens",
              flush=True)
        res = spawn({"model_dir": str(model_dir), "items": items, "n": args.samples, "max_tokens": args.max_tokens,
                     "out": str(out / f"{tag}.jsonl"), "gpu_memory_utilization": args.gpu_mem,
                     "max_num_seqs": args.max_num_seqs}, out, tag)
        if not res.get("ok"):
            print(json.dumps(res, indent=1))
            raise SystemExit(1)
        rows = [json.loads(line) for line in open(out / f"{tag}.jsonl")]
        rows_by_model[spec] = rows
        results[spec] = {"overall": summarize(rows), "s": res["s"],
                         "by_set": {src: summarize([r for r in rows if r["source"] == src])
                                    for src in sorted({r["source"] for r in rows})}}
        o = results[spec]["overall"]
        print(f"       acc {o['accuracy']:.3f}  cut {o['cut_share']:.2f}  mean tokens {o['mean_tokens']}  "
              f"({res['s']:.0f} s)", flush=True)
    if len(args.models) >= 2:
        a, b = args.models[0], args.models[1]
        results["paired"] = {"original": a, "edited": b, **paired(rows_by_model[a], rows_by_model[b])}
    (out / "results.json").write_text(json.dumps(results, indent=1))

    print(f"\n{'model':<28}{'set':<10}{'acc':>7}{'cut':>7}{'tokens':>9}")
    for spec in args.models:
        for src, s in results[spec]["by_set"].items():
            print(f"{spec[:27]:<28}{src:<10}{s['accuracy']:>7.3f}{s['cut_share']:>7.2f}{s['mean_tokens']:>9}")
    if "paired" in results:
        p = results["paired"]
        print(f"\npaired ({p['edited']} - {p['original']}): {p['diff']:+.3f} +/- {p['se']:.3f} over {p['items']} "
              f"items; wrong->right {p['wrong_to_right']}, right->wrong {p['right_to_wrong']}")
    print(f"results: {out / 'results.json'}")


if __name__ == "__main__":
    main()
