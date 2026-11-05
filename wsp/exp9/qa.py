"""Short-answer questions for Experiment 9: load a model (optionally with a LoRA adapter), answer greedily with
thinking off, grade against targets.

Uses Hugging Face transformers rather than vLLM: answers are a few tokens, so vLLM's minutes-long engine startup
would dominate. A model loads in seconds and answers 100 questions in seconds.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

INSTRUCTION = "Answer with only the answer itself (a name, number, place or short phrase). No explanation."


def norm_compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_correct(output: str, target: str) -> bool:
    """Target found in the output (punctuation/case-insensitive). For a person, the family name is enough."""
    out = norm_compact(output)
    if norm_compact(target) in out:
        return True
    parts = target.split()
    return len(parts) == 2 and len(parts[1]) > 3 and norm_compact(parts[1]) in out


def messages(question: str, context: str | None = None) -> list[dict]:
    system = INSTRUCTION + (f"\n\nReference information:\n{context}" if context else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


class Answerer:
    def __init__(self, model_dir: str, adapter: str | None = None, quant: str | None = None, edit_dir=None):
        import torch
        from transformers import AutoTokenizer

        from ..exp8.teacher_forced import load_model

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = load_model(model_dir, quant, edit_dir)
        from ..exp8.common import profile_for
        self.batch = profile_for(model_dir).get("qa_batch", 32)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()

    @classmethod
    def from_model(cls, model, tok):
        """Wrap an already loaded model (e.g. one being edited in memory)."""
        import torch

        a = cls.__new__(cls)
        a.torch, a.model, a.tok = torch, model, tok
        a.tok.padding_side = "left"
        if a.tok.pad_token is None:
            a.tok.pad_token = a.tok.eos_token
        return a

    def prompt(self, msgs: list[dict]) -> str:
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def answer(self, convs: list[list[dict]], max_new_tokens: int = 24, batch: int | None = None) -> list[str]:
        batch = batch or getattr(self, "batch", 32)
        outs = []
        for i in range(0, len(convs), batch):
            texts = [self.prompt(m) for m in convs[i:i + batch]]
            # inputs go where the embedding table is (CPU for 4-bit models, GPU otherwise)
            dev = self.model.get_input_embeddings().weight.device
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                txt = self.tok.decode(row, skip_special_tokens=True)
                outs.append(txt.split("</think>")[-1].strip())
        return outs


def run(req: dict) -> dict:
    """Worker entry for the agent's forward-pass tool: {"model_dir", "questions": [...], "out"}."""
    t0 = time.time()
    a = Answerer(req["model_dir"], req.get("adapter"), req.get("quant"), req.get("edit_dir"))
    qs = req["questions"]
    outs = a.answer([messages(q) for q in qs], max_new_tokens=int(req.get("max_new_tokens", 24)))
    rows = [{"question": q, "answer": o} for q, o in zip(qs, outs)]
    Path(req["out"]).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {"ok": True, "type": "qa", "file": req["out"], "answers": rows, "s": round(time.time() - t0, 1)}


def spawn(req: dict, log_path: Path) -> dict:
    import subprocess

    rp = Path(str(req["out"]) + ".request.json")
    rp.write_text(json.dumps(req))
    with open(log_path, "w") as log:
        p = subprocess.run([sys.executable, "-m", "wsp.exp9.qa", str(rp)], stdout=log, stderr=subprocess.STDOUT,
                           cwd=Path(__file__).resolve().parents[2])
    res = Path(str(req["out"]) + ".result.json")
    if p.returncode != 0 or not res.exists():
        return {"ok": False, "error": f"qa pass failed (exit {p.returncode})", "log_tail": log_path.read_text()[-3000:]}
    return json.loads(res.read_text())


if __name__ == "__main__":
    req = json.loads(Path(sys.argv[1]).read_text())
    result = run(req)
    Path(str(req["out"]) + ".result.json").write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "answers"}))
