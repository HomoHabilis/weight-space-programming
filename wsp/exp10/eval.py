"""Score a model on LiveCodeBench: generate (vLLM, fresh process), then run the tests in a sandbox.

  python -m wsp.exp10.eval --model base9b --set test --name base9b-nothink
  python -m wsp.exp10.eval --model base9b --edit-dir runs/agent/<run>/edit --set test --name <run>
  python -m wsp.exp10.eval --model base4b --set test --thinking --max-tokens 32768 --limit 20 --name speed-4b-think

Writes runs/exp10_eval/<name>.json (summary) and <name>.rows.jsonl (one graded sample per line).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

from .. import RUNS
from . import generate, lcb

OUT = RUNS / "exp10_eval"


def pick_ids(set_name: str, limit: int | None, seed: int = 0, difficulties: list[str] | None = None) -> list[str] | None:
    """A fixed subset (by difficulty label only, then stratified sampling), so smaller runs stay comparable."""
    probs = [p for p in lcb.load(set_name) if not difficulties or p["difficulty"] in difficulties]
    if not limit:
        return [p["id"] for p in probs] if difficulties else None
    by = collections.defaultdict(list)
    for p in probs:
        by[p["difficulty"]].append(p["id"])
    rng, total = random.Random(seed), sum(len(v) for v in by.values())
    ids = []
    for d in sorted(by):
        k = max(1, round(limit * len(by[d]) / total))
        ids += rng.sample(sorted(by[d]), min(k, len(by[d])))
    return ids[:limit]


def summarize(rows: list[dict]) -> dict:
    def rate(rs):
        return round(sum(r["passed"] for r in rs) / max(1, len(rs)), 4)
    by_diff = collections.defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    why = collections.Counter(r.get("why", "passed") for r in rows)
    return {"pass@1": rate(rows), "samples": len(rows), "problems": len({r["id"] for r in rows}),
            "by_difficulty": {d: rate(v) for d, v in sorted(by_diff.items())},
            "no_code": sum(r["code"] is None for r in rows), "unfinished": sum(not r["finished"] for r in rows),
            "mean_tokens": round(sum(r["n_tokens"] for r in rows) / max(1, len(rows))), "outcomes": dict(why)}


def evaluate(model: str, set_name: str = "test", edit_dir: str | None = None, thinking: bool = False,
             max_tokens: int = 6144, n: int = 1, limit: int | None = None, name: str = "eval",
             max_num_seqs: int = 16, kv_fp8: bool = False, out_dir: Path = OUT, think_budget: int | None = None,
             answer_tokens: int = 4096, difficulties: list[str] | None = None) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_file = out_dir / f"{name}.gen.jsonl"
    req = {"model": model, "edit_dir": edit_dir, "set": set_name, "ids": pick_ids(set_name, limit, difficulties=difficulties),
           "thinking": thinking, "max_tokens": max_tokens, "n": n, "out": str(gen_file),
           "max_num_seqs": max_num_seqs, "kv_fp8": kv_fp8, "think_budget": think_budget,
           "answer_tokens": answer_tokens}
    meta = generate.spawn(req, out_dir / f"{name}.gen.log")
    if not meta.get("ok"):
        raise RuntimeError(meta.get("error", "") + "\n" + meta.get("log_tail", ""))
    gens = [json.loads(line) for line in open(gen_file)]
    probs = {p["id"]: p for p in lcb.load(set_name)}
    t1 = time.time()
    results = lcb.grade_many([(probs[g["id"]], g["code"]) for g in gens])
    rows = [{**{k: v for k, v in g.items() if k != "text"}, **res} for g, res in zip(gens, results)]
    with open(out_dir / f"{name}.rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"name": name, "model": model, "edit_dir": edit_dir, "set": set_name, "thinking": thinking,
               "max_tokens": max_tokens, "think_budget": think_budget, "answer_tokens": answer_tokens if think_budget else None,
               "kv_fp8": kv_fp8, "n": n, "limit": limit, "difficulties": difficulties, **summarize(rows),
               "forced_stop": sum(bool(r.get("forced_stop")) for r in rows),
               "timing_s": {"build": meta["s_build"], "load": meta["s_load"], "generate": meta["s_gen"],
                            "grade": round(time.time() - t1, 1), "total": round(time.time() - t0, 1)},
               "tok_per_s": meta["tok_per_s"]}
    (out_dir / f"{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="base9b")
    ap.add_argument("--edit-dir")
    ap.add_argument("--set", default="test", choices=["test", "practice"])
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--limit", type=int, help="fixed difficulty-stratified subset of this many problems")
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--kv-fp8", action="store_true")
    ap.add_argument("--think-budget", type=int, help="with --thinking: stop thinking after this many tokens, then answer")
    ap.add_argument("--answer-tokens", type=int, default=4096, help="answer length after a thinking budget")
    ap.add_argument("--difficulty", help="comma-separated labels to keep, e.g. easy,medium")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    s = evaluate(a.model, a.set, a.edit_dir, a.thinking, a.max_tokens, a.n, a.limit, a.name, a.max_num_seqs, a.kv_fp8,
                 think_budget=a.think_budget, answer_tokens=a.answer_tokens,
                 difficulties=a.difficulty.split(",") if a.difficulty else None)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
