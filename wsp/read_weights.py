"""Experiments 1 and 2: read a network's weights and say what it does. No tools, one reply.

Sends the authors' exact prompts (data/lm_understand-logs/exp*/runs/<name>/prompt.txt) to a backend and scores the
replies the way the authors did (keywords on the 'Game:' / 'Images:' line, open answer).

  python -m wsp.read_weights run   --exp 1 --backend anthropic --effort high
  python -m wsp.read_weights run   --exp 2 --backend copilot --model gpt-6-astra
  python -m wsp.read_weights score --exp 1 --published          # score the authors' Astra answers (sanity check)
  python -m wsp.read_weights score --exp 1 --runs runs/exp1/anthropic-claude-opus-5-5-high
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import LOGS, RUNS
from .backends import dump_json, make_chat

EXPS = {
    1: {"folder": "exp1_which_game", "field": "Game", "key": "game",
        "keywords": {  # Frogger counts as Freeway, as in the paper
            "breakout": ["breakout", "brick", "arkanoid"],
            "pong": ["pong"],
            "pacman": ["pac-man", "pacman", "pac man", "ms. pac"],
            "freeway": ["freeway", "frogger", "road-crossing", "road crossing", "traffic-crossing", "crossy"]}},
    2: {"folder": "exp2_diffusion", "field": "Images", "key": "shape",
        "keywords": {
            "circle": ["circle", "circular", "disc", "disk", "round", "oval", "ring", "ellipse"],
            "square": ["square", "rectangle", "rectangular", "box"],
            "triangle": ["triangle", "triangular", "pyramid", "wedge"]}},
}
NAMES = "Observations|Reasoning|Game|Description|Confidence|Images|Example"
# 'Game: x', '**Game:** x' or a markdown heading '## Game'
HEADER = r"^\s*(?:#+\s*)?(?:\*\*)?(?:{})(?:\*\*)?\s*(?::|$)\**\s*(.*)$"
SECTION = re.compile(HEADER.format(NAMES), re.I)


def answer_line(text: str, field: str) -> str:
    """The text of the 'Field:' section (same line, or the following lines up to the next section)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(HEADER.format(field), line, re.I)
        if m:
            parts = [m.group(1)] if m.group(1).strip() else []
            for nxt in lines[i + 1:]:
                if SECTION.match(nxt):
                    break
                if nxt.strip():
                    parts.append(nxt.strip())
                if parts and not nxt.strip():
                    break
            return " ".join(parts).strip()
    return ""


def classify(line: str, keywords: dict[str, list[str]]) -> str | None:
    """The class whose keyword appears first in the line."""
    low, best = line.lower(), None
    for cls, words in keywords.items():
        for w in words:
            j = low.find(w)
            if j >= 0 and (best is None or j < best[0]):
                best = (j, cls)
    return best[1] if best else None


def confidence(text: str) -> int | None:
    m = re.search(r"Confidence\W*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def published_runs(exp: int) -> dict[str, Path]:
    runs = LOGS / EXPS[exp]["folder"] / "runs"
    if not runs.exists():
        sys.exit(f"{runs} not found: run scripts/fetch_logs.sh first")
    return {p.name: p for p in sorted(runs.iterdir()) if (p / "prompt.txt").exists()}


def answer_key(exp: int) -> dict[str, str]:
    """Run name -> true class. Run names in the release are the true class (e.g. runs/breakout), the answer key maps
    network ids to the class; we use both and check they agree."""
    key = json.loads((LOGS / EXPS[exp]["folder"] / "eval" / "answer_key.json").read_text())
    classes = {v[EXPS[exp]["key"]] for v in key.values()}
    out = {}
    for name in published_runs(exp):
        assert name in classes, f"run {name} is not a class name in the answer key {sorted(classes)}"
        out[name] = name
    return out


def score(exp: int, runs_dir: Path) -> dict:
    truth, rows = answer_key(exp), []
    for name in truth:
        f = runs_dir / name / "answer.md"
        if not f.exists():
            continue
        text = f.read_text()
        line = answer_line(text, EXPS[exp]["field"])
        guess = classify(line, EXPS[exp]["keywords"])
        rows.append({"run": name, "true": truth[name], "guess": guess, "correct": guess == truth[name],
                     "confidence": confidence(text), "answer_line": line[:200]})
    n = sum(r["correct"] for r in rows)
    return {"exp": exp, "runs_dir": str(runs_dir), "correct": n, "total": len(rows), "rows": rows}


def run(args) -> Path:
    tag = f"{args.backend}-{args.model or 'default'}-{args.effort}".replace("/", "_")
    out_root = RUNS / f"exp{args.exp}" / tag
    for name, src in published_runs(args.exp).items():
        if args.only and name not in args.only:
            continue
        out = out_root / name
        if (out / "answer.md").exists() and not args.force:
            print(f"[skip] {name}: already answered ({out / 'answer.md'})")
            continue
        out.mkdir(parents=True, exist_ok=True)
        prompt = (src / "prompt.txt").read_text()
        (out / "prompt.txt").write_text(prompt)
        print(f"[run] exp{args.exp} {name}: {len(prompt):,} chars -> {args.backend}", flush=True)
        chat = make_chat(args.backend, args.model, args.effort, out, fallback=args.fallback, dry_run=args.dry_run)
        reply = chat.send(prompt)
        dump_json(out / "run.json", reply.meta)
        if args.dry_run:
            print(json.dumps(reply.meta, indent=1))
            continue
        (out / "answer.md").write_text(reply.text)
        line = answer_line(reply.text, EXPS[args.exp]["field"])
        print(f"      -> {EXPS[args.exp]['field']}: {line[:120]!r}  ({reply.meta.get('wall_s')} s)")
    return out_root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exp", type=int, choices=[1, 2], required=True)
    r.add_argument("--backend", choices=["claude-code", "copilot", "anthropic", "manual"], default="claude-code")
    r.add_argument("--model", help="default: claude-opus-5-5 (claude-code, anthropic) / gpt-6-astra (copilot)")
    r.add_argument("--effort", default="high", help="the paper used high for experiments 1 and 2")
    r.add_argument("--only", nargs="*", help="run names, e.g. breakout pong")
    r.add_argument("--force", action="store_true", help="re-ask runs that already have an answer")
    r.add_argument("--fallback", action="store_true", help="anthropic: enable server-side refusal fallback")
    r.add_argument("--dry-run", action="store_true", help="copilot: print the command instead of running it")
    s = sub.add_parser("score")
    s.add_argument("--exp", type=int, choices=[1, 2], required=True)
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--published", action="store_true", help="score the authors' Astra answers")
    g.add_argument("--runs", type=Path, help="a folder written by `run`")
    args = ap.parse_args()

    if args.cmd == "run":
        out_root = run(args)
        if not args.dry_run:
            res = score(args.exp, out_root)
            dump_json(out_root / "scores.json", res)
            print_scores(res)
    else:
        runs_dir = LOGS / EXPS[args.exp]["folder"] / "runs" if args.published else args.runs
        print_scores(score(args.exp, runs_dir))


def print_scores(res: dict) -> None:
    for r in res["rows"]:
        mark = "OK " if r["correct"] else "BAD"
        print(f"  {mark} {r['run']:<9} guess={str(r['guess']):<9} conf={r['confidence']}  {r['answer_line'][:90]}")
    print(f"exp{res['exp']}: {res['correct']}/{res['total']} correct  ({res['runs_dir']})")


if __name__ == "__main__":
    main()
