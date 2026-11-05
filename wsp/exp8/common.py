"""Shared pieces of Experiment 8: model files, problems, prompts, grading, building an edited checkpoint."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .. import LOGS, ROOT

MODELS = ROOT / "models"
BASE_REPO, BASE_REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"  # revision used in the paper
EDITED_REPO = "RikiyaT/Qwen3.5-0.8B-weight-space-programming"  # Astra's 50-pass edit, published by the authors
PRACTICE = LOGS / "exp8_math" / "practice"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
# The model card's thinking-mode settings, used for every number in the paper.
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}
FROZEN = ("model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight")


def download(repo: str, revision: str | None = None) -> Path:
    """A local folder with real files (not cache symlinks), so it can be bind-mounted into the sandbox."""
    from huggingface_hub import snapshot_download

    out = MODELS / repo.split("/")[-1]
    if not (out / "config.json").exists():
        snapshot_download(repo, revision=revision, local_dir=out)
    return out


def resolve_model(spec: str) -> Path:
    """'base', 'published-edit', a local folder, or a Hugging Face repo id."""
    if spec == "base":
        return download(BASE_REPO, BASE_REV)
    if spec == "published-edit":
        return download(EDITED_REPO)
    if Path(spec).exists():
        return Path(spec)
    return download(spec)


# ---------------------------------------------------------------- problems
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def practice_problems() -> dict[str, dict]:
    out = {}
    for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        for r in load_jsonl(PRACTICE / name):
            out[r["id"]] = r
    return out


def eval_set(name: str, limit: int | None = None, seed: int = 0) -> list[dict]:
    """Benchmark problems (never shown to the editor model). A fixed random subset if limit is set."""
    import random

    from datasets import load_dataset

    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"id": f"math500_{i}", "source": "math500", "problem": r["problem"], "answer": r["answer"]}
                for i, r in enumerate(ds)]
    elif name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [{"id": f"gsm8k_test_{i}", "source": "gsm8k", "problem": r["question"],
                 "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(ds)]
    else:
        raise ValueError(name)
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
    return rows


def user_message(problem: str) -> str:
    return f"{problem}\n{INSTRUCTION}"


def item_seed(item_id: str, sample: int) -> int:
    """Sampling seed fixed by the problem, so original and edited models share their random numbers."""
    return int(hashlib.sha256(f"{item_id}:{sample}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- grading
def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...}, brace matched (as in hendrycks/math)."""
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        rest = text[idx + len("\\boxed"):].strip()
        return rest.split("$")[0].split("\n")[0].strip() or None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def answer_part(text: str) -> str | None:
    """The text after the reasoning; None if the reasoning never closed."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    return None if "<think>" in text else text


def grade(text: str, gold: str, finished: bool) -> dict:
    from math_verify import parse, verify

    ans = answer_part(text)
    pred = last_boxed(ans) if ans is not None else None
    ok = False
    if pred is not None:
        try:
            ok = bool(verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}")))
        except Exception:
            ok = pred.strip() == str(gold).strip()
    if ok:
        failure = None
    elif not finished:
        failure = "cut_at_max_tokens"
    elif pred is None:
        failure = "no_answer_finished"
    else:
        failure = "wrong_answer"
    return {"pred": pred, "correct": ok, "failure": failure}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"generations": 0}
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r.get("source", "?"), []).append(r["correct"])
    fails: dict[str, int] = {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 0}
    for r in rows:
        if r["failure"]:
            fails[r["failure"]] += 1
    return {"generations": n, "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
            "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in by_src.items()},
            "mean_tokens": round(sum(r["n_tokens"] for r in rows) / n),
            "cut_share": round(sum(not r["finished"] for r in rows) / n, 4), "failures": fails}


def paired(orig: list[dict], edit: list[dict]) -> dict:
    """Edited minus original accuracy over items, with its standard error and flip counts."""
    import statistics

    def per_item(rows):
        d: dict[str, list] = {}
        for r in rows:
            d.setdefault(r["id"], []).append(r["correct"])
        return {k: sum(v) / len(v) for k, v in d.items()}

    a, b = per_item(orig), per_item(edit)
    ids = sorted(set(a) & set(b))
    diffs = [b[i] - a[i] for i in ids]
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float("nan")
    ko = {(r["id"], r["sample"]): r["correct"] for r in orig}
    ke = {(r["id"], r["sample"]): r["correct"] for r in edit}
    both = set(ko) & set(ke)
    return {"items": len(ids), "diff": round(sum(diffs) / max(len(diffs), 1), 4), "se": round(se, 4),
            "wrong_to_right": sum(1 for k in both if ke[k] and not ko[k]),
            "right_to_wrong": sum(1 for k in both if ko[k] and not ke[k])}


# ---------------------------------------------------------------- edited checkpoints
def build_edited(base: Path, edit_dir: Path, out: Path) -> Path:
    """Copy of the base model folder with the tensors in edit_dir/<tensor name>.npy replaced (cast to bfloat16)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    edits = {p.stem: p for p in edit_dir.glob("*.npy")}
    stamp = hashlib.sha256("".join(f"{k}:{p.stat().st_mtime_ns}" for k, p in sorted(edits.items())).encode())
    stamp = stamp.hexdigest()[:16]
    if (out / ".edit_stamp").exists() and (out / ".edit_stamp").read_text() == stamp:
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    used = set()
    for f in base.iterdir():
        if f.suffix == ".safetensors":
            tensors = load_file(f)
            for name in list(tensors):
                if name in edits:
                    new = torch.from_numpy(np.load(edits[name]).astype(np.float32))
                    assert tuple(new.shape) == tuple(tensors[name].shape), (name, new.shape, tensors[name].shape)
                    tensors[name] = new.to(tensors[name].dtype).contiguous()
                    used.add(name)
            save_file(tensors, out / f.name, metadata={"format": "pt"})
        elif f.is_file():
            os.link(f, out / f.name) if os.stat(f).st_dev == os.stat(out).st_dev else shutil.copy2(f, out / f.name)
    missing = set(edits) - used
    if missing:
        raise ValueError(f"edited tensors not in the model: {sorted(missing)[:5]}")
    (out / ".edit_stamp").write_text(stamp)
    return out
