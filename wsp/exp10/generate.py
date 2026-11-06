"""Generation worker: one vLLM instance writes solutions for a list of LiveCodeBench problems, then exits.

  python -m wsp.exp10.generate <request.json>

request: {"model": "base4b" | "base9b" | <folder>, "edit_dir": null | <folder of name.npy edits>,
          "set": "practice" | "test", "ids": null | [...], "thinking": false, "max_tokens": 6144, "n": 1,
          "out": "<file>.jsonl", "max_num_seqs": 16, "max_model_len": null, "kv_fp8": false}
Writes one JSON line per sample (with the extracted code, not yet graded) and <out>.meta.json.

Models larger than the GPU are quantized to 4-bit (MXFP4) while loading; every MLP/attention block that holds an
edited matrix is kept in bfloat16 so 4-bit rounding cannot erase the edit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from .. import ROOT
from ..exp8.common import profile_for, resolve_model
from . import lcb

# Qwen3.5 model card: precise coding in thinking mode / instruct (non-thinking) mode.
SAMPLING = {True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
            False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}}
CACHE = ROOT / "models" / "edited"
# Qwen3's documented early-exit sentence for a thinking budget
EARLY_STOP = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."


def prior_edits(folder) -> list[str]:
    """Tensor names already edited in a model folder written by edited_dir (a stacked starting point), else []."""
    f = Path(folder) / ".done"
    return json.loads(f.read_text()) if f.exists() else []


def edited_dir(base: Path, edit_dir: Path, out: Path | None = None) -> Path:
    """A folder that looks like the base checkpoint with the edits written in. Shards without edits are hard links,
    so only the shards holding an edited tensor are rewritten (a few GB, not 19)."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    edits = {p.stem: p for p in Path(edit_dir).glob("*.npy")}
    stamp = hashlib.sha256("".join(f"{k}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for k, p in sorted(edits.items()))
                           .encode() + str(base.resolve()).encode()).hexdigest()[:16]
    out = Path(out) if out else CACHE / f"{base.name}-{stamp}"
    if (out / ".done").exists() and (out / ".stamp").exists() and (out / ".stamp").read_text() == stamp:
        return out
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    used = set()
    for f in sorted(base.iterdir()):
        if not f.is_file() or f.name.startswith("."):  # never hard-link markers (writing them would change base's)
            continue
        if f.suffix == ".safetensors":
            with safe_open(f, "pt") as h:
                names = set(h.keys())
            if names & set(edits):
                tensors = load_file(f)
                for name in names & set(edits):
                    new = torch.from_numpy(np.load(edits[name]).astype(np.float32))
                    assert tuple(new.shape) == tuple(tensors[name].shape), name
                    tensors[name] = new.to(tensors[name].dtype).contiguous()
                    used.add(name)
                save_file(tensors, out / f.name, metadata={"format": "pt"})
                del tensors
                continue
        os.link(f, out / f.name)
    if set(edits) - used:
        raise ValueError(f"edited tensors not in the model: {sorted(set(edits) - used)[:5]}")
    (out / ".stamp").write_text(stamp)
    (out / ".done").write_text(json.dumps(sorted(set(edits) | set(prior_edits(base)))))
    return out


def keep_bf16_patterns(edit_names) -> list[str]:
    """vLLM layer patterns for the blocks holding edited matrices (fused layers must be kept whole)."""
    pats = set()
    for n in edit_names:
        m = re.search(r"layers\.(\d+)\.(mlp|self_attn|linear_attn)\.", n)
        if m:
            pats.add(f"*layers.{m.group(1)}.{m.group(2)}.*")
    return sorted(pats)


def run(req: dict) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    base = resolve_model(req["model"])
    prof = profile_for(req["model"]) or profile_for(base)
    model_dir, edits = base, []
    if req.get("edit_dir"):
        edits = sorted(p.stem for p in Path(req["edit_dir"]).glob("*.npy"))
        if edits:
            model_dir = edited_dir(base, Path(req["edit_dir"]), req.get("edited_model_dir"))
    t_build = time.time() - t0
    thinking, n, max_tokens = bool(req.get("thinking")), int(req.get("n", 1)), int(req.get("max_tokens", 6144))
    probs = lcb.load(req.get("set", "test"))
    if req.get("ids"):
        keep = set(req["ids"])
        probs = [p for p in probs if p["id"] in keep]
    tok = AutoTokenizer.from_pretrained(base)
    prompts = [tok.apply_chat_template(lcb.messages(p), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=thinking) for p in probs]
    longest = max(len(tok(p).input_ids) for p in prompts)
    budget = int(req.get("think_budget") or 0) if thinking else 0
    answer_tokens = int(req.get("answer_tokens", 4096))
    total = budget + answer_tokens + 64 if budget else max_tokens
    kw = dict(model=str(model_dir), tokenizer=str(base), dtype="bfloat16", seed=0, enable_prefix_caching=True,
              max_model_len=int(req.get("max_model_len") or longest + total + 16),
              gpu_memory_utilization=float(req.get("gpu_memory_utilization", 0.80)),
              max_num_seqs=int(req.get("max_num_seqs", 16)), language_model_only=True,
              disable_log_stats=False)  # logs real generation throughput every few seconds
    bf16_blocks = keep_bf16_patterns(set(edits) | set(prior_edits(base)))
    if prof.get("quant"):  # larger than the GPU in bfloat16
        # blocks edited here or in the starting point (a stacked edit) stay bfloat16
        kw.update(quantization="online", quantization_config={"linear": "mxfp4", "ignore": bf16_blocks})
    if req.get("kv_fp8"):
        kw["kv_cache_dtype"] = "fp8"
    lora = None
    if req.get("lora_dir"):  # a LoRA adapter (bfloat16) on top of the 4-bit model, as QLoRA is normally served
        from vllm.lora.request import LoRARequest
        rank = json.loads((Path(req["lora_dir"]) / "adapter_config.json").read_text())["r"]
        kw.update(enable_lora=True, max_lora_rank=max(8, rank), max_loras=1)
        lora = LoRARequest("adapter", 1, str(req["lora_dir"]))
    llm = LLM(**kw)
    t_load = time.time() - t0 - t_build
    rows = []
    if budget:
        # Thinking budget (Qwen's recipe): think for at most `budget` tokens; if the model has not closed its
        # thinking by then, tell it to stop and answer. Then write the answer (at most `answer_tokens`).
        sp1 = SamplingParams(max_tokens=budget, n=n, seed=0, stop=["</think>"], **SAMPLING[True])
        first = llm.generate(prompts, sp1, lora_request=lora)
        cont, info = [], []
        for p, prompt, o in zip(probs, prompts, first):
            for k, c in enumerate(o.outputs):
                forced = c.finish_reason != "stop"
                think = c.text + ("\n\n" + EARLY_STOP if forced else "") + "\n</think>\n\n"
                cont.append(prompt + think)
                info.append((p, k, think, len(c.token_ids), forced))
        second = llm.generate(cont, SamplingParams(max_tokens=answer_tokens, seed=1, **SAMPLING[True]),
                              lora_request=lora)
        for (p, k, think, n_think, forced), o in zip(info, second):
            c = o.outputs[0]
            rows.append((p, k, think + c.text, n_think + len(c.token_ids), c.finish_reason == "stop",
                         {"think_tokens": n_think, "answer_tokens": len(c.token_ids), "forced_stop": forced}))
    else:
        outs = llm.generate(prompts, SamplingParams(max_tokens=max_tokens, n=n, seed=0, **SAMPLING[thinking]),
                            lora_request=lora)
        for p, o in zip(probs, outs):
            for k, c in enumerate(o.outputs):
                rows.append((p, k, c.text, len(c.token_ids), c.finish_reason == "stop", {}))
    t_gen = time.time() - t0 - t_build - t_load
    out = Path(req["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(out, "w") as f:
        for p, k, text, ntok, finished, extra in rows:
            n_tok += ntok
            f.write(json.dumps({"id": p["id"], "sample": k, "difficulty": p["difficulty"], "platform": p["platform"],
                                "n_tokens": ntok, "finished": finished, **extra,
                                "code": lcb.extract_code(text), "text": text}) + "\n")
    meta = {"ok": True, "model": req["model"], "model_dir": str(model_dir), "edits": edits, "bf16_blocks": bf16_blocks,
            "thinking": thinking,
            "lora_dir": req.get("lora_dir"),
            "think_budget": budget or None, "answer_tokens": answer_tokens if budget else None,
            "set": req.get("set", "test"), "problems": len(probs), "n": n, "max_tokens": max_tokens,
            "tokens": n_tok, "s_build": round(t_build, 1), "s_load": round(t_load, 1), "s_gen": round(t_gen, 1),
            "tok_per_s": round(n_tok / max(t_gen, 1e-9), 1)}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def spawn(req: dict, log: Path) -> dict:
    """Run one request in a fresh process (vLLM frees the GPU only when its process exits)."""
    import subprocess

    log.parent.mkdir(parents=True, exist_ok=True)
    rp = log.with_suffix(".request.json")
    rp.write_text(json.dumps(req))
    import signal
    import threading

    if threading.current_thread() is threading.main_thread():  # turn `kill` into an exception so `finally` runs
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with open(log, "w") as fh:
        # own process group, so the vLLM engine (a grandchild) is stopped too if this run is stopped
        p = subprocess.Popen([sys.executable, "-m", "wsp.exp10.generate", str(rp)], stdout=fh,
                             stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
        try:
            p.wait()
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    mp = Path(str(req["out"]) + ".meta.json")
    if p.returncode != 0 or not mp.exists():
        return {"ok": False, "error": f"generation failed (exit {p.returncode})", "log_tail": log.read_text()[-3000:]}
    return json.loads(mp.read_text())


if __name__ == "__main__":
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), indent=1))
