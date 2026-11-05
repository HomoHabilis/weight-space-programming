"""Copy the small, shareable outputs of runs/ (git-ignored) into results/ (tracked), so results can be committed.

  python -m wsp.collect_results            # then: git add results && git commit

Kept: summaries, evaluation answers, agent transcripts (prompts, replies, code and outputs, notes, final answer,
forward-pass requests and summaries), edit manifests. Left out: model weights, edit tensors, activation arrays,
generated .jsonl texts of large passes and anything over 5 MB. Local paths are replaced by placeholders.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import ROOT, RUNS

OUT = ROOT / "results"
TEXT = {".json", ".jsonl", ".md", ".txt", ".log"}
MAX_BYTES = 5 * 2**20
SKIP_DIRS = {"adapter", "edit", "edited", "manual"}  # weights / tensors
SKIP_PREFIXES = ("fwd_", "gen_", "best_")             # activation dumps and big generations inside work/


def scrub(text: str) -> str:
    for path, tag in ((str(ROOT), "<repo>"), (str(Path.home()), "<home>")):
        text = text.replace(path, tag)
    return text


def keep(rel: Path) -> bool:
    parts = rel.parts
    if any(p in SKIP_DIRS for p in parts[:-1]):
        return False
    if "work" in parts and any(p.startswith(SKIP_PREFIXES) for p in parts[parts.index("work") + 1:]):
        return False
    if "requests" in parts and rel.suffix == ".jsonl":
        return False
    return rel.suffix in TEXT


def main():
    copied, skipped = 0, 0
    for src in sorted(RUNS.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(RUNS)
        if not keep(rel) or src.stat().st_size > MAX_BYTES:
            skipped += 1
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.write_text(scrub(src.read_text()))
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
        copied += 1
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{copied} files copied to {OUT.relative_to(ROOT)} ({size / 2**20:.1f} MB), {skipped} skipped")
    leaks = [str(f) for f in OUT.rglob("*") if f.is_file() and f.suffix in TEXT
             and (os.environ.get("USER", "\0") in f.read_text(errors="ignore") or str(Path.home()) in
                  f.read_text(errors="ignore"))]
    if leaks:
        print("WARNING: local user name or home path still present in:", *leaks[:10], sep="\n  ")


if __name__ == "__main__":
    main()
