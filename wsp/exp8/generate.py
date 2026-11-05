"""Generation worker: one vLLM instance answers a list of problems, then exits (frees the GPU).

  python -m wsp.exp8.generate <request.json>

request.json: {"model_dir": ..., "items": [{"id", "source", "user", "answer"}], "n": 1, "max_tokens": 32768,
               "out": "<file>.jsonl", "gpu_memory_utilization": 0.85, "max_num_seqs": 32}
Writes one JSON line per generation (graded) to `out` and a summary to `<out>.summary.json`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .common import SAMPLING, grade, item_seed, summarize


def run(req: dict) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    model_dir, n, max_tokens = req["model_dir"], int(req.get("n", 1)), int(req.get("max_tokens", 32768))
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompts, meta = [], []
    for it in req["items"]:
        text = tok.apply_chat_template([{"role": "user", "content": it["user"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
        for k in range(n):
            prompts.append(text)
            meta.append((it, k))
    longest = max(len(tok(p).input_ids) for p in prompts)
    llm_kw = dict(model=model_dir, dtype="bfloat16", max_model_len=longest + max_tokens + 16,
                  gpu_memory_utilization=float(req.get("gpu_memory_utilization", 0.85)),
                  max_num_seqs=int(req.get("max_num_seqs", 32)), seed=0, enable_prefix_caching=True)
    try:  # Qwen3.5 checkpoints include a vision tower: skip it, we only use text
        llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **llm_kw)
    except (TypeError, ValueError):
        llm = LLM(**llm_kw)
    params = [SamplingParams(max_tokens=max_tokens, seed=item_seed(it["id"], k), **SAMPLING) for it, k in meta]
    outs = llm.generate(prompts, params)
    rows = []
    for (it, k), o in zip(meta, outs):
        c = o.outputs[0]
        text = "<think>" + c.text if "<think>" in prompts[0][-20:] else c.text  # template may open the think tag
        finished = c.finish_reason == "stop"
        g = grade(text, it["answer"], finished)
        rows.append({"id": it["id"], "source": it.get("source"), "sample": k, "n_tokens": len(c.token_ids),
                     "finished": finished, **g, "text": c.text})
    out = Path(req["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {"ok": True, "file": str(out), **summarize(rows), "s": round(time.time() - t0, 1)}
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def spawn(req: dict, workdir: Path, name: str) -> dict:
    """Run one generation request in a fresh process (vLLM releases the GPU only when its process exits)."""
    import subprocess

    workdir.mkdir(parents=True, exist_ok=True)
    rp = workdir / f"{name}.request.json"
    rp.write_text(json.dumps(req))
    with open(workdir / f"{name}.log", "w") as log:
        p = subprocess.run([sys.executable, "-m", "wsp.exp8.generate", str(rp)], stdout=log, stderr=subprocess.STDOUT,
                           cwd=Path(__file__).resolve().parents[2])
    sp = Path(str(req["out"]) + ".summary.json")
    if p.returncode != 0 or not sp.exists():
        tail = (workdir / f"{name}.log").read_text()[-3000:]
        return {"ok": False, "error": f"generation failed (exit {p.returncode})", "log_tail": tail}
    return json.loads(sp.read_text())


if __name__ == "__main__":
    print(json.dumps(run(json.loads(Path(sys.argv[1]).read_text())), indent=1))
