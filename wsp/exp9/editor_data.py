"""What the editor model may see in Experiment 9 (mounted read-only as /practice in its sandbox).

Only the fact statements, their 3 train phrasings and answers, and general questions it may use to check that
unrelated answers stay the same. Never the held-out phrasings, the evaluation's neighbor company, the validation
sets or the evaluation's real-world questions.
"""
from __future__ import annotations

import json

from .eval import load
from .make_data import OUT

EDITOR_DIR = OUT / "editor"


def write() -> str:
    from .rank_one import GENERAL_PROTECT, PRESERVE_COMPANIES

    EDITOR_DIR.mkdir(parents=True, exist_ok=True)
    with open(EDITOR_DIR / "facts.jsonl", "w") as f:
        for x in load("facts.jsonl"):
            f.write(json.dumps({"id": x["id"], "kind": x["kind"], "statement": x["statement"], "answer": x["answer"],
                                "questions": x["train"]}) + "\n")
    with open(EDITOR_DIR / "general.jsonl", "w") as f:
        for q, a in GENERAL_PROTECT:
            f.write(json.dumps({"question": q, "answer": a}) + "\n")
    (EDITOR_DIR / "other_companies.json").write_text(json.dumps(PRESERVE_COMPANIES))
    return str(EDITOR_DIR)
