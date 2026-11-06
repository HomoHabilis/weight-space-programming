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


# Target models for weight editing. quant: how it is loaded on an 8 GB GPU (None = bfloat16, "nf4" = 4-bit).
PROFILES = {
    "base": {"repo": BASE_REPO, "rev": BASE_REV, "quant": None,                       # Qwen3.5-0.8B
             "layers": 24, "d_model": 1024, "d_mlp": 3584,
             "description": "Qwen3.5-0.8B (text part: 24 layers of width 1,024; layers 3, 7, 11, 15, 19 and 23 are "
                            "full-attention layers with 8 query heads and 2 key/value heads of size 256 and a gated "
                            "output, the other 18 are Gated DeltaNet linear-attention layers with 16 heads of size "
                            "128; each layer has a SwiGLU MLP of width 3,584; RMSNorm; vocabulary 248,320; input "
                            "embedding and output layer tied)"},
    "base2b": {"repo": "Qwen/Qwen3.5-2B", "rev": None, "quant": None},
    "base4b": {"repo": "Qwen/Qwen3.5-4B", "rev": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "quant": "nf4",
               "qa_batch": 16, "layers": 32, "d_model": 2560, "d_mlp": 9216,
               "description": "Qwen3.5-4B (text part: 32 layers of width 2,560; layers 3, 7, 11, 15, 19, 23, 27 and "
                              "31 are full-attention layers with 16 query heads and 4 key/value heads of size 256 and "
                              "a gated output, the other 24 are Gated DeltaNet linear-attention layers with 16 key "
                              "heads and 32 value heads of size 128; each layer has a SwiGLU MLP of width 9,216; "
                              "RMSNorm; vocabulary 248,320; input embedding and output layer tied). It is larger than "
                              "the GPU allows at full precision, so the forward-pass tool runs it in 4-bit (NF4); every "
                              "matrix you edit is kept in bfloat16 there, so your edits are applied exactly. The "
                              "weights you read in the kernel are the original bfloat16 weights. The vocabulary matrix "
                              "(248,320 x 2,560) is 2.5 GB in float32: read the rows you need with "
                              "load_rows(name, rows) instead of load(name)"},
    "base9b": {"repo": "Qwen/Qwen3.5-9B", "rev": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "quant": "nf4",
               "qa_batch": 8,  # 9B in 4-bit uses 5.7 GB; a batch of 32 short answers peaked at the 8 GB limit
               "layers": 32, "d_model": 4096, "d_mlp": 12288,
               "description": "Qwen3.5-9B (text part: 32 layers of width 4,096; layers 3, 7, 11, 15, 19, 23, 27 and "
                              "31 are full-attention layers with 16 query heads and 4 key/value heads of size 256 and "
                              "a gated output, the other 24 are Gated DeltaNet linear-attention layers with 16 key "
                              "heads and 32 value heads of size 128; each layer has a SwiGLU MLP of width 12,288; "
                              "RMSNorm; vocabulary 248,320; input embedding and output layer separate (lm_head.weight)). "
                              "It is larger than the GPU, so the forward-pass tool runs it in 4-bit (NF4); every matrix "
                              "you edit is kept in bfloat16 there, so your edits are applied exactly. The weights you "
                              "read in the kernel are the original bfloat16 weights. Its two vocabulary matrices "
                              "(248,320 x 4,096) are 4 GB each in float32: read the rows you need with "
                              "load_rows(name, rows) instead of load(name)"},
}


def profile_for(path_or_spec) -> dict:
    """The profile of a model spec or of a local model folder (matched by folder name); {} if unknown."""
    if str(path_or_spec) in PROFILES:
        return PROFILES[str(path_or_spec)]
    name = Path(str(path_or_spec)).name
    return next((p for p in PROFILES.values() if p["repo"].split("/")[-1] == name), {})


def resolve_model(spec: str) -> Path:
    """'base' (0.8B), 'base2b', 'base4b', 'base9b', 'published-edit', a local folder, or a Hugging Face repo id."""
    if spec in PROFILES:
        return download(PROFILES[spec]["repo"], PROFILES[spec]["rev"])
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
