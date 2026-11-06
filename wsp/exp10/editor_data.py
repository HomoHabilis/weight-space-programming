"""What the editor model may see in Experiment 10 (mounted read-only as /practice in its sandbox).

Only practice problems (LiveCodeBench release v5 additions, easy + medium, contests before 2025-01-04) with all their
tests, the unedited model's own answers to them (the same samples the QLoRA baseline trains on), and the exact
generation settings. Never a scorecard problem (release v6, 2025-01-04 and later).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path

from .. import ROOT, RUNS
from . import generate, lcb

EDITOR_DIR = ROOT / "data" / "exp10" / "editor"
SAMPLES = RUNS / "exp10_eval" / "samples-4b-think8k-n3"
DIFFICULTIES = ("easy", "medium")
SETTINGS = {
    "chat_template": "the model's own (/model/tokenizer_config.json / chat_template.jinja), enable_thinking=True",
    "system": lcb.SYSTEM, "user_stdin": "### Question:\n{question}\n\n" + lcb.FMT_STDIN
    + "### Answer: (use the provided format with backticks)\n\n",
    "user_call": "### Question:\n{question}\n\n" + lcb.FMT_CALL + "### Answer: (use the provided format with backticks)\n\n",
    "think_budget": 8192, "answer_tokens": 4096,
    "procedure": "1) generate up to think_budget tokens after the prompt (which ends with '<think>\\n'), stopping at "
                 "'</think>'; 2) if thinking did not end by itself, append '\\n\\n' + early_stop; then append "
                 "'\\n</think>\\n\\n'; 3) generate the answer, up to answer_tokens tokens. The last ```python block "
                 "of the answer is run against all tests (6 s per test).",
    "early_stop": generate.EARLY_STOP,
    "sampling": generate.SAMPLING[True], "weights": "4-bit (MXFP4) except blocks holding edited matrices (bfloat16)",
}


def write(samples: Path = SAMPLES) -> str:
    EDITOR_DIR.mkdir(parents=True, exist_ok=True)
    tdir = EDITOR_DIR / "tests"
    tdir.mkdir(exist_ok=True)
    probs = [p for p in lcb.load("practice") if p["difficulty"] in DIFFICULTIES]
    ids = {p["id"] for p in probs}
    with open(EDITOR_DIR / "problems.jsonl", "w") as f:
        for p in probs:
            f.write(json.dumps({k: p[k] for k in ("id", "title", "platform", "difficulty", "date", "question",
                                                 "starter", "func", "public", "n_tests")}) + "\n")
            src, dst = lcb.PREP / "tests" / f"{p['id']}.json.gz", tdir / f"{p['id']}.json.gz"
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
    gen, rows = Path(f"{samples}.gen.jsonl"), Path(f"{samples}.rows.jsonl")
    if gen.exists() and rows.exists():
        graded = {(r["id"], r["sample"]): r for r in map(json.loads, open(rows))}
        with open(EDITOR_DIR / "samples.jsonl", "w") as f:
            for g in map(json.loads, open(gen)):
                if g["id"] not in ids:
                    continue
                r = graded[(g["id"], g["sample"])]
                f.write(json.dumps({"id": g["id"], "sample": g["sample"], "difficulty": g["difficulty"],
                                    "passed": r["passed"], "outcome": r.get("why", "passed"),
                                    "think_tokens": g.get("think_tokens"), "answer_tokens": g.get("answer_tokens"),
                                    "forced_stop": g.get("forced_stop"), "text": g["text"]}) + "\n")
    (EDITOR_DIR / "settings.json").write_text(json.dumps(SETTINGS, indent=1))
    (EDITOR_DIR / "README.txt").write_text(
        "problems.jsonl  practice problems (statement, starter code, public examples, number of tests)\n"
        "tests/<id>.json.gz  all tests of a problem: gzip JSON list of {input, output, testtype}\n"
        "samples.jsonl   the unedited model's answers to these problems (3 per problem, full text incl. thinking),\n"
        "                graded on all tests: passed, outcome (passed / no code / wrong / error / timeout)\n"
        "settings.json   exactly how answers are generated and graded\n")
    return str(EDITOR_DIR)


def summary() -> dict:
    s = [json.loads(line) for line in open(EDITOR_DIR / "samples.jsonl")]
    from collections import Counter
    return {"samples": len(s), "passed": sum(x["passed"] for x in s), "outcomes": dict(Counter(x["outcome"] for x in s))}


if __name__ == "__main__":
    print(write())
    if (EDITOR_DIR / "samples.jsonl").exists():
        print(summary())
