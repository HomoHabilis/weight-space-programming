"""Experiment 8 harness: a frontier model edits Qwen3.5-0.8B's weights to make it better at math.

The model replies with exactly one ```python block (run in the sandboxed kernel) or one ```forward block (a JSON
request run on the local GPU), as in the paper. The run ends when the forward-pass budget is spent and the model
replies without a block. Defaults are scaled down from the paper to fit one 8 GB GPU.

  python -m wsp.exp8.agent --backend anthropic --effort high --passes 10 --name opus-p10
  python -m wsp.exp8.agent --backend copilot --model gpt-6-astra --effort xhigh --passes 10 --name astra-p10
  python -m wsp.exp8.agent --resume runs/agent/opus-p10          # continue after a crash (new conversation)

Afterwards: python -m wsp.exp8.eval_lite --models base models/edited/<name> --sets math500:100 gsm8k:100
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from .. import ROOT, RUNS
from ..backends import make_chat
from . import generate as G
from . import teacher_forced as TF
from .common import MODELS, PRACTICE, build_edited, practice_problems, resolve_model, user_message
from .sandbox import Kernel

BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)```", re.S)
OUT_LIMIT = 10_000
# Opus 5.5 list prices, $ per million tokens (for a running estimate only)
PRICE = {"input_tokens": 4.0, "output_tokens": 20.0, "cache_read_input_tokens": 0.20,
         "cache_creation_input_tokens": 5.0}


def _prompt_fields(target: str) -> dict:
    from .common import PROFILES, profile_for
    prof = profile_for(target) or PROFILES["base"]  # a model folder (e.g. a stacked edit) is matched by folder name
    return {"model_description": prof["description"], "n_layers": prof["layers"],
            "hs_kb": round(2 * prof["d_model"] / 1024), "mlp_kb": round(2 * prof["d_mlp"] / 1024)}


def _start_fields(target: str) -> dict:
    """Code task, no thinking: the prompt says whether the starting weights already hold an earlier edit."""
    from ..exp10.generate import prior_edits
    if prior_edits(resolve_model(target)):  # a stacked starting point (models/stacked/...)
        return {"starting_model": "starting model",
                "start_note": " These weights already contain an earlier direct edit (made to improve the model with "
                              "thinking enabled); it is part of the model you start from: /model, load_original(name) "
                              "and \"original\" in the forward-pass tool all mean these weights.",
                "keep_note": " or that the earlier edit changed, which stay in bfloat16. Your edit must not undo what "
                             "the earlier edit achieved with thinking enabled: that mode is scored too."}
    return {"starting_model": "unedited model", "start_note": "",
            "keep_note": ", which stay in bfloat16. Thinking-enabled mode is scored too: your edit should not make it "
                         "worse."}


def task_prompt(cfg: dict) -> str:
    name = cfg.get("prompt") or ("exp9_task.md" if cfg.get("task") == "facts" else "exp8_task.md")
    text = (ROOT / "prompts" / name).read_text()
    for k, v in cfg.items():
        text = text.replace("{" + k + "}", f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else str(v))
    return text


def ascii_cut(s: str, limit: int = OUT_LIMIT) -> str:
    s = s.encode("ascii", "replace").decode()
    return s if len(s) <= limit else s[:limit] + f"\n[output cut at {limit:,} characters]"


class Harness:
    def __init__(self, run_dir: Path, cfg: dict, args):
        self.dir, self.cfg, self.args = run_dir, cfg, args
        self.work = run_dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        (run_dir / "turns").mkdir(exist_ok=True)
        (run_dir / "requests").mkdir(exist_ok=True)
        self.state_path = run_dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "turns": 0, "code_runs": 0, "passes_used": 0, "conversations": 0, "done": False, "usage": {}}
        from .common import profile_for
        self.model_dir = resolve_model(cfg.get("target", "base"))
        self.four_bit = profile_for(self.model_dir).get("quant") == "nf4"  # edits applied at load time, no copies
        self.edited_dir = MODELS / "edited" / run_dir.name
        if cfg.get("task") == "facts":
            from ..exp9.editor_data import write
            practice_dir, self.practice = Path(write()), {}
        elif cfg.get("task") == "code":
            from ..exp10 import editor_data as ED
            practice_dir = Path(ED.write(Path(cfg["samples"]) if cfg.get("samples") else ED.SAMPLES,
                                         thinking=cfg.get("thinking", True)))
            self.practice = {json.loads(line)["id"] for line in open(practice_dir / "problems.jsonl")}
        else:
            practice_dir, self.practice = PRACTICE, practice_problems()
        self.kernel = Kernel(self.model_dir, practice_dir, self.work, mem_gb=cfg["kernel_mem_gb"],
                             sandbox=not args.no_sandbox)

    # ------------------------------------------------------------ bookkeeping
    def save_state(self):
        self.state_path.write_text(json.dumps(self.state, indent=1))

    def log_tool(self, rec: dict):
        with open(self.dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def pending_notices(self) -> str:
        """Notes from the experimenters: files dropped in <run>/notices/ are delivered once, at the start of the next
        message (as the paper's authors did during their runs), and repeated in every later handoff."""
        d = self.dir / "notices"
        sent = self.state.setdefault("notices_delivered", [])
        new = [f for f in sorted(d.glob("*.txt")) if f.name not in sent] if d.exists() else []
        for f in new:
            sent.append(f.name)
        return "".join(f"NOTE FROM THE EXPERIMENTERS: {f.read_text().strip()}\n\n" for f in new)

    def all_notices(self) -> str:
        d = self.dir / "notices"
        return "".join(f"NOTE FROM THE EXPERIMENTERS: {(d / n).read_text().strip()}\n\n"
                       for n in self.state.get("notices_delivered", []) if (d / n).exists())

    def host_to_sandbox(self, obj):
        return json.loads(json.dumps(obj).replace(str(self.work), "/work"))

    def cost(self) -> float:
        """API-equivalent dollars at Opus 5.5 list prices. 1-hour cache writes (Claude Code uses them) cost 2x input."""
        u = self.state["usage"]
        base = sum(u.get(k, 0) * p / 1e6 for k, p in PRICE.items())
        return base + u.get("cache_write_1h_tokens", 0) * (8.0 - PRICE["cache_creation_input_tokens"]) / 1e6

    # ------------------------------------------------------------ conversation
    def new_chat(self, first_message: str):
        self.state["conversations"] += 1
        self.conv_turns = 0
        self.chat = make_chat(self.args.backend, self.args.model, self.args.effort, self.dir,
                              fallback=self.args.fallback)
        return first_message

    def handoff_message(self, pending: str) -> str:
        notes = (self.work / "notes.md").read_text() if (self.work / "notes.md").exists() else "(none)"
        edits = sorted(p.stem for p in (self.work / "edited").glob("*.npy")) if (self.work / "edited").exists() else []
        s = self.state
        return (task_prompt(self.cfg) + "\n\n" + self.all_notices() + "THIS IS A NEW CONVERSATION continuing a session already in progress "
                f"(the earlier conversation was closed to keep the context short). Used so far: {s['passes_used']} of "
                f"{self.cfg['passes']} forward passes, {s['code_runs']} of {self.cfg['max_code_runs']} code runs. "
                "The kernel may have been restarted (reload your variables). Your files in /work are kept.\n\n"
                f"YOUR NOTES (/work/notes.md):\n{notes[:20000]}\n\nCURRENT EDIT (/work/edited): "
                f"{', '.join(edits) if edits else 'none'}\n\nLAST MESSAGE OF THE PREVIOUS CONVERSATION:\n{pending}")

    def needs_handoff(self, meta: dict) -> bool:
        if self.cfg["handoff_turns"] and self.conv_turns >= self.cfg["handoff_turns"]:
            return True
        u = meta.get("usage") or {}
        ctx = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return ctx > self.cfg["handoff_tokens"]

    # ------------------------------------------------------------ tools
    def run_python(self, code: str) -> str:
        if self.state["code_runs"] >= self.cfg["max_code_runs"]:
            return "[code budget exhausted: no more Python runs are possible; use your forward passes or finish]"
        self.state["code_runs"] += 1
        t0 = time.time()
        out, err, restarted = self.kernel.execute(code, timeout=self.cfg["kernel_timeout"])
        self.log_tool({"kind": "python", "run": self.state["code_runs"], "code": code, "output": out, "error": err,
                       "restarted": restarted, "s": round(time.time() - t0, 1)})
        return (f"Output of code run {self.state['code_runs']} of {self.cfg['max_code_runs']}"
                f"{' (error)' if err else ''}:\n```\n{ascii_cut(out) or '(no output)'}\n```")

    def run_forward(self, body: str) -> str:
        cfg, s = self.cfg, self.state
        if s["passes_used"] >= cfg["passes"]:
            return "[all forward passes are used: make your final choice and reply without a code block]"
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            return f"[forward request rejected (not counted): invalid JSON: {e}]"
        k = s["passes_used"] + 1
        (self.dir / "requests" / f"fwd_{k}.json.claimed").write_text(json.dumps(req, indent=1))
        try:
            res = self.forward(req, k)
        except ValueError as e:
            res = {"ok": False, "error": str(e)}
        self.log_tool({"kind": "forward", "pass": k, "request": req, "result": res})
        if res.get("ok"):
            s["passes_used"] = k
            (self.dir / "requests" / f"fwd_{k}.done.json").write_text(json.dumps(res, indent=1))
            head = f"Forward pass {k} of {cfg['passes']} done."
        else:
            head = "Forward request failed (not counted against the budget)."
        res = self.host_to_sandbox({x: y for x, y in res.items() if x != "log_tail"} | (
            {"log_tail": res["log_tail"][-1500:]} if not res.get("ok") and "log_tail" in res else {}))
        return f"{head}\n```json\n{ascii_cut(json.dumps(res, indent=1))}\n```"

    def model_for(self, which: str) -> Path:
        if which == "original":
            return self.model_dir
        edits = list((self.work / "edited").glob("*.npy")) if (self.work / "edited").exists() else []
        if not edits:
            raise ValueError("there is no edit yet (save_edit something first)")
        return build_edited(self.model_dir, self.work / "edited", self.edited_dir)

    def model_req(self, which: str) -> dict:
        """What a GPU worker loads: an edited copy for small models; for 4-bit models the base plus the edit folder,
        applied at load time (writing a 19 GB copy of a 9B model per pass would be wasteful)."""
        if self.four_bit:
            if which != "original" and not any((self.work / "edited").glob("*.npy")):
                raise ValueError("there is no edit yet (save_edit something first)")
            return {"model_dir": str(self.model_dir), "edit_dir": None if which == "original" else str(self.work / "edited")}
        return {"model_dir": str(self.model_for(which))}

    def read_work_file(self, path) -> list:
        """A list the editor saved in /work (JSON list, or JSON Lines), for requests too long to write inline."""
        if not isinstance(path, str) or not path.startswith("/work/") or ".." in path:
            raise ValueError("files must be given as /work/<name>")
        f = self.work / path[len("/work/"):]
        if not f.is_file():
            raise ValueError(f"{path} not found")
        text = f.read_text()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not isinstance(data, list):
            raise ValueError(f"{path} must hold a JSON list (or one JSON value per line)")
        return data

    def tf_size(self, texts: list, ret: dict) -> tuple[int, float]:
        """(texts x longest text in tokens, GB of returned arrays) for a teacher-forced request. Chats are counted as
        their contents plus a few template tokens per message (close to the chat template's length)."""
        if not hasattr(self, "_tok"):
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        def n_tokens(x):
            if isinstance(x, list):  # a chat
                return sum(len(self._tok.encode(str(m.get("content", ""))).ids) + 6 for m in x) + 8
            return len(self._tok.encode(str(x)).ids)

        longest = min(max(n_tokens(x) for x in texts), self.cfg["tf_max_tokens_per_text"])
        padded = len(texts) * longest
        from .common import profile_for
        prof = profile_for(self.model_dir)
        d_model, d_mlp = prof.get("d_model", 1024), prof.get("d_mlp", 3584)
        per_token = (2 * d_model * len(ret.get("hidden_states") or []) + 2 * d_mlp * len(ret.get("mlp_act") or [])
                     + 4 * bool(ret.get("token_logprobs", True)) + 6 * int(ret.get("topk") or 0) + 4)
        return padded, padded * per_token / 2**30

    def forward(self, req: dict, k: int) -> dict:
        if self.cfg.get("file_inputs"):
            for key in ("questions", "texts", "chats", "problems"):  # problems: code task
                if f"{key}_file" in req:
                    req[key] = self.read_work_file(req.pop(f"{key}_file"))
        cfg, kind, which = self.cfg, req.get("type"), req.get("model")
        facts_task = cfg.get("task") == "facts"
        if cfg.get("task") == "code":
            if kind == "code":
                return self.code_pass(req, k)
            if kind != "teacher_forced":
                raise ValueError('"type" must be "code" or "teacher_forced"')
        if kind == "qa" and facts_task:
            from ..exp9 import qa as QA
            if which not in ("original", "edited"):
                raise ValueError('"model" must be "original" or "edited"')
            qs = req.get("questions") or []
            if not 1 <= len(qs) <= cfg["qa_max"] or not all(isinstance(q, str) for q in qs):
                raise ValueError(f"questions must be a list of 1 to {cfg['qa_max']} strings")
            print(f"    [gpu] qa pass: {len(qs)} questions ({which})...", flush=True)
            return QA.spawn({**self.model_req(which), "questions": qs,
                             "out": str(self.work / f"qa_{k}.jsonl")}, self.dir / "requests" / f"fwd_{k}.log")
        if kind == "generate" and facts_task:
            raise ValueError('this task has "qa" and "teacher_forced" passes')
        if kind == "generate":
            if which not in ("original", "edited", "both"):
                raise ValueError('"model" must be "original", "edited" or "both"')
            probs = req.get("problems") or []
            missing = [p for p in probs if p not in self.practice]
            if missing:
                raise ValueError(f"unknown practice ids: {missing[:5]}")
            n = int(req.get("n", 1))
            cap = cfg["max_gens_half"] if which == "both" else cfg["max_gens"]
            if not probs or len(probs) * n > cap:
                raise ValueError(f"problems x n must be between 1 and {cap} (got {len(probs) * n})")
            mt = int(req.get("max_tokens", cfg["gen_max_tokens"]))
            if not 1 <= mt <= cfg["gen_max_tokens"]:
                raise ValueError(f"max_tokens must be at most {cfg['gen_max_tokens']}")
            items = [{"id": p, "source": self.practice[p]["source"], "user": user_message(self.practice[p]["problem"]),
                      "answer": self.practice[p]["answer"]} for p in probs]
            models = ["original", "edited"] if which == "both" else [which]
            out = {"ok": True, "type": "generate", "model": which}
            rows = {}
            for m in models:
                name = f"gen_{k}" + (f"_{m}" if which == "both" else "")
                print(f"    [gpu] generating {len(items)}x{n} with the {m} model (max {mt} tokens)...", flush=True)
                r = G.spawn({"model_dir": str(self.model_for(m)), "items": items, "n": n, "max_tokens": mt,
                             "out": str(self.work / f"{name}.jsonl"), "gpu_memory_utilization": cfg["gpu_mem"],
                             "max_num_seqs": cfg["max_num_seqs"]}, self.dir / "requests", name)
                if not r.get("ok"):
                    return r
                (self.work / f"{name}.jsonl.summary.json").unlink(missing_ok=True)
                out[m] = r
                rows[m] = [json.loads(line) for line in open(self.work / f"{name}.jsonl")]
            if which == "both":
                from .common import paired
                out["paired"] = paired(rows["original"], rows["edited"])
            else:
                out = {**out, **out.pop(which)}
            return out
        if kind == "teacher_forced":
            if which not in ("original", "edited"):
                raise ValueError('"model" must be "original" or "edited"')
            texts = req.get("texts") or req.get("chats") or []
            if not 1 <= len(texts) <= cfg["tf_max_texts"]:
                raise ValueError(f"between 1 and {cfg['tf_max_texts']} texts")
            ret = req.get("return") or {"token_logprobs": True}
            if len(ret.get("hidden_states") or []) > 4 or len(ret.get("mlp_act") or []) > 4:
                raise ValueError("at most 4 layers each for hidden_states and mlp_act")
            if int(ret.get("topk") or 0) > 20:
                raise ValueError("topk at most 20")
            size = ""
            if cfg.get("tf_max_padded_tokens"):  # v3: limit by size, not by number of texts
                padded, gb = self.tf_size(texts, ret)
                if padded > cfg["tf_max_padded_tokens"] or gb > cfg["tf_max_output_gb"]:
                    raise ValueError(f"too large: {len(texts)} texts x longest text = {padded:,} tokens (limit "
                                     f"{cfg['tf_max_padded_tokens']:,}), returned arrays about {gb:.1f} GB (limit "
                                     f"{cfg['tf_max_output_gb']} GB). Split it, or ask for fewer layers")
                size = f", {padded:,} padded tokens, ~{gb:.2f} GB"
            print(f"    [gpu] teacher-forced pass on {len(texts)} texts ({which}{size})...", flush=True)
            return TF.spawn({**self.model_req(which), "out_dir": str(self.work / f"fwd_{k}"),
                             "texts": req.get("texts"), "chats": req.get("chats"),
                             "enable_thinking": req.get("enable_thinking", False),
                             "max_tokens_per_text": cfg["tf_max_tokens_per_text"], "return": ret},
                            self.dir / "requests" / f"fwd_{k}.log")
        raise ValueError('"type" must be ' + ('"qa" or "teacher_forced"' if facts_task else
                                              '"generate" or "teacher_forced"'))

    def code_pass(self, req: dict, k: int) -> dict:
        """Experiment 10: the model writes solutions to practice problems (thinking budget), graded on all tests."""
        from ..exp10 import eval as E10
        cfg, which = self.cfg, req.get("model")
        if which not in ("original", "edited"):
            raise ValueError('"model" must be "original" or "edited"')
        probs, n = req.get("problems") or [], int(req.get("n", 1))
        if not isinstance(probs, list) or not all(isinstance(p, str) for p in probs):
            raise ValueError('"problems" must be a list of practice problem ids')
        missing = [p for p in probs if p not in self.practice]
        if missing:
            raise ValueError(f"unknown practice ids: {missing[:5]}")
        if len(set(probs)) != len(probs) or not 1 <= n <= 3 or not 1 <= len(probs) * n <= cfg["code_max"]:
            raise ValueError(f"distinct problems, n in 1..3, problems x n between 1 and {cfg['code_max']}")
        used = sum(1 for f in self.work.glob("code_*.json") if re.fullmatch(r"code_\d+\.json", f.name))
        if used >= cfg["max_code_passes"]:
            raise ValueError(f"all {cfg['max_code_passes']} code passes are used (teacher_forced passes remain)")
        edit_dir = None
        if which == "edited":
            if not any((self.work / "edited").glob("*.npy")):
                raise ValueError("there is no edit yet (save_edit something first)")
            edit_dir = str(self.work / "edited")
        print(f"    [gpu] code pass: {len(probs)} problems x {n} ({which})...", flush=True)
        try:
            thinking = cfg.get("thinking", True)
            s = E10.evaluate(cfg["target"], "practice", edit_dir, thinking=thinking, n=n, name=f"code_{k}",
                             max_num_seqs=12, kv_fp8=True, out_dir=self.work, max_tokens=cfg.get("max_tokens", 12288),
                             think_budget=cfg["think_budget"] if thinking else None,
                             answer_tokens=cfg["answer_tokens"], ids=probs,
                             edited_model_dir=str(self.dir / "edited_model"))
        except RuntimeError as e:
            return {"ok": False, "error": "code pass failed", "log_tail": str(e)[-3000:]}
        rows = [json.loads(line) for line in open(self.work / f"code_{k}.rows.jsonl")]
        (self.work / f"code_{k}.gen.log").unlink(missing_ok=True)
        return {"ok": True, "type": "code", "model": which, "pass@1": s["pass@1"], "by_difficulty": s["by_difficulty"],
                "outcomes": s["outcomes"], "forced_stop": s["forced_stop"], "minutes": round(s["timing_s"]["total"] / 60, 1),
                "results": [{"id": r["id"], "sample": r["sample"], "passed": r["passed"], "outcome": r.get("why", "passed"),
                             **({"think_tokens": r.get("think_tokens"), "answer_tokens": r.get("answer_tokens")}
                                if thinking else {"tokens": r.get("n_tokens"), "finished": r.get("finished")})}
                            for r in rows],
                "files": {"texts": f"/work/code_{k}.gen.jsonl", "graded": f"/work/code_{k}.rows.jsonl"}}

    def send_with_retry(self, msg: str):
        """Usage limits (e.g. a Claude subscription's 5-hour window) or network errors: wait and resend the same
        turn instead of failing. Gives up after --max-wait-hours."""
        waited = 0.0
        while True:
            try:
                return self.chat.send(msg)
            except Exception as e:  # noqa: BLE001 - any backend error is retried the same way
                if waited >= self.args.max_wait_hours * 3600:
                    raise
                delay = 600 if re.search(r"limit|rate|quota|overloaded|529|429", str(e), re.I) else 120
                print(f"[wait] backend error: {str(e)[:200]!r}; retrying in {delay // 60} min "
                      f"(waited {waited / 3600:.1f} h so far)", flush=True)
                time.sleep(delay)
                waited += delay

    # ------------------------------------------------------------ main loop
    def run(self):
        s = self.state
        print(self.kernel.start() or "[kernel ready]", flush=True)
        msg = self.new_chat(task_prompt(self.cfg) if s["turns"] == 0 else self.handoff_message(
            "(the harness was restarted; continue where you left off)"))
        while not s["done"] and s["turns"] < self.cfg["max_turns"]:
            s["turns"] += 1
            self.conv_turns += 1
            tdir = self.dir / "turns" / f"t{s['turns']:03d}"
            tdir.mkdir(parents=True, exist_ok=True)
            msg = self.pending_notices() + msg
            (tdir / "prompt.txt").write_text(msg)
            reply = self.send_with_retry(msg)
            (tdir / "answer.md").write_text(reply.text)
            (tdir / "run.json").write_text(json.dumps(reply.meta, indent=1, default=str))
            for k_, v in (reply.meta.get("usage") or {}).items():
                if isinstance(v, int):
                    s["usage"][k_] = s["usage"].get(k_, 0) + v
            cw = ((reply.meta.get("usage") or {}).get("cache_creation") or {}).get("ephemeral_1h_input_tokens")
            if isinstance(cw, int):
                s["usage"]["cache_write_1h_tokens"] = s["usage"].get("cache_write_1h_tokens", 0) + cw
            blocks = BLOCK.findall(reply.text)
            note = ""
            if len(blocks) > 1:
                note = f"[your reply had {len(blocks)} code blocks: only the first was run]\n"
            if not blocks:
                if s["passes_used"] >= self.cfg["passes"]:
                    s["done"] = True
                    (self.dir / "final_answer.md").write_text(reply.text)
                    break
                msg = (f"[no code block in your reply: the session continues until all {self.cfg['passes']} forward "
                       f"passes are used ({s['passes_used']} used so far). Reply with one ```python or ```forward "
                       "block.]")
            elif blocks[0][0] == "python":
                msg = note + self.run_python(blocks[0][1])
            else:
                msg = note + self.run_forward(blocks[0][1])
            if s["passes_used"] >= self.cfg["passes"] and "final choice" not in msg:
                msg += ("\n\nAll forward passes are used. Make your final choice (the tensors in /work/edited are "
                        "submitted) and reply without a code block.")
            kind = blocks[0][0] if blocks else "text"
            print(f"[turn {s['turns']}] {kind:<7} passes {s['passes_used']}/{self.cfg['passes']}  code runs "
                  f"{s['code_runs']}  ~${self.cost():.2f} API-equiv  ({reply.meta.get('wall_s')} s)", flush=True)
            self.save_state()
            if self.needs_handoff(reply.meta):
                print("[handoff] starting a new conversation with the task, notes and current edit", flush=True)
                msg = self.new_chat(self.handoff_message(msg))
        self.save_state()
        self.kernel.stop()
        if (self.work / "edited").exists() and any((self.work / "edited").glob("*.npy")):
            shutil.copytree(self.work / "edited", self.dir / "edit", dirs_exist_ok=True)
            if self.cfg.get("task") == "code":
                out = self.dir / "edit"
                mode = (f"--thinking --think-budget {self.cfg['think_budget']} --answer-tokens {self.cfg['answer_tokens']}"
                        if self.cfg.get("thinking", True) else f"--max-tokens {self.cfg.get('max_tokens', 12288)}")
                how = (f"python -m wsp.exp10.eval --model {self.cfg.get('target')} --edit-dir {out.relative_to(ROOT)} "
                       f"--set test --difficulty easy,medium {mode} --kv-fp8 --max-num-seqs 12 --name {self.dir.name}")
            elif self.four_bit:
                out = self.dir / "edit"
                how = (f"python -m wsp.exp9.eval --model {self.cfg.get('target')} --edit-dir {out.relative_to(ROOT)} "
                       f"--name {self.dir.name}")
            else:
                out = build_edited(self.model_dir, self.work / "edited", self.edited_dir)
                how = (f"python -m wsp.exp9.eval --model {out.relative_to(ROOT)}" if self.cfg.get("task") == "facts"
                       else f"python -m wsp.exp8.eval_lite --models base {out.relative_to(ROOT)} --sets math500:100 "
                            f"gsm8k:100")
            print(f"\nDone. Edit: {out}\nEvaluate: {how}")
        else:
            print("\nDone, with no edit submitted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["claude-code", "copilot", "anthropic", "manual"], default="claude-code")
    ap.add_argument("--model", help="default: claude-opus-5-5 (claude-code, anthropic) / gpt-6-astra (copilot)")
    ap.add_argument("--effort", default=None, help="default: high (math), xhigh (facts). The paper used max.")
    ap.add_argument("--name", help="run folder name under runs/agent/")
    ap.add_argument("--resume", type=Path, help="continue an existing run folder")
    ap.add_argument("--target", default="base",
                    help="target model to edit: base (Qwen3.5-0.8B), base2b, base4b, base9b (4-bit on the GPU)")
    ap.add_argument("--task", choices=["math", "facts", "code"], default="math",
                    help="math: Experiment 8 (paper). facts: Experiment 9 (write company facts into the weights). "
                         "code: Experiment 10 (LiveCodeBench with a thinking budget)")
    ap.add_argument("--max-code-passes", type=int, default=8, help="code task: passes that generate code")
    ap.add_argument("--code-max", type=int, default=40, help="code task: problems x samples per code pass")
    ap.add_argument("--think-budget", type=int, default=8192, help="code task: thinking budget")
    ap.add_argument("--answer-tokens", type=int, default=4096, help="code task: answer length after thinking")
    ap.add_argument("--no-thinking", action="store_true", help="code task: thinking off (answers of --max-tokens)")
    ap.add_argument("--max-tokens", type=int, default=12288, help="code task, no thinking: answer length")
    ap.add_argument("--samples", help="code task: the starting model's practice samples (a runs/exp10_eval/<name> "
                                      "prefix); default: the base model's thinking samples")
    ap.add_argument("--base-score", default="about 55%", help="code task: the starting model's score, for the prompt")
    ap.add_argument("--passes", type=int, default=None, help="forward-pass budget (math default 10, facts 30)")
    ap.add_argument("--qa-max", type=int, default=400, help="facts task: questions per qa pass")
    ap.add_argument("--harness", choices=["v2", "v3"], default="v3",
                    help="facts task harness version (docs/exp9_company_knowledge.md). v3: teacher-forced passes "
                         "limited by size (texts x longest text) instead of 128 texts")
    ap.add_argument("--max-gens", type=int, default=128, help="generations per pass (paper: 512)")
    ap.add_argument("--gen-max-tokens", type=int, default=16384, help="per answer in passes (paper: 32768)")
    ap.add_argument("--eval-max-tokens", type=int, default=32768, help="stated in the prompt for the final eval")
    ap.add_argument("--tf-max-texts", type=int, default=None, help="default: 128 (math, facts v2), 1024 (facts v3)")
    ap.add_argument("--tf-max-tokens-per-text", type=int, default=8192)
    ap.add_argument("--max-code-runs", type=int, default=None, help="default: 300 (math, as in the paper), 600 (facts)")
    ap.add_argument("--max-turns", type=int, default=None, help="default: 400 (math), 800 (facts)")
    ap.add_argument("--kernel-timeout", type=int, default=900)
    ap.add_argument("--kernel-mem-gb", type=float, default=10)
    ap.add_argument("--handoff-turns", type=int, default=None,
                    help="new conversation every N turns; 0 = off. Default: 40 (math, as in the paper), 0 (facts)")
    ap.add_argument("--handoff-tokens", type=int, default=None,
                    help="new conversation when the context exceeds this. Default: 500k (math), 300k (facts)")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-wait-hours", type=float, default=8, help="keep retrying a failing backend this long "
                    "(subscription usage limits reset every few hours)")
    ap.add_argument("--fallback", action="store_true", help="anthropic: server-side refusal fallback (off: purity)")
    ap.add_argument("--no-sandbox", action="store_true", help="run the kernel without bubblewrap (not recommended)")
    args = ap.parse_args()

    if args.resume:
        run_dir = args.resume.resolve()
        cfg = json.loads((run_dir / "config.json").read_text())
        saved = cfg.pop("_args")
        args = argparse.Namespace(**{**saved, "resume": run_dir})  # same backend, model and settings as before
    else:
        facts = args.task in ("facts", "code")  # code shares the facts task's (v3) defaults
        if args.task == "code":
            args.harness = "v3"
            if args.passes is None:
                args.passes = 24
            if args.tf_max_tokens_per_text == 8192:
                args.tf_max_tokens_per_text = 16384  # thinking traces are long
        defaults = {"passes": (30, 10), "effort": ("xhigh", "high"), "max_code_runs": (600, 300),
                    "max_turns": (800, 400), "handoff_turns": (0, 40), "handoff_tokens": (300_000, 500_000)}
        v3 = facts and args.harness == "v3"
        defaults["tf_max_texts"] = (1024 if v3 else 128, 128)
        for key, (f_val, m_val) in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, f_val if facts else m_val)
        name = args.name or time.strftime(f"{args.task}-{args.backend}-p{args.passes}-%Y%m%d-%H%M")
        run_dir = RUNS / "agent" / name
        if (run_dir / "state.json").exists():
            raise SystemExit(f"{run_dir} exists: use --resume {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"task": args.task, "qa_max": args.qa_max, "passes": args.passes,
               # v2 harness for the facts task: prompt with file inputs (prompts/exp9_task_v2.md)
               "harness": args.harness if facts else "paper",
               "target": args.target,
               # model-specific parts of the v3 task prompt
               **_prompt_fields(args.target),
               "prompt": (("exp10_task_nothink.md" if args.no_thinking else "exp10_task.md") if args.task == "code" else
                          {"v2": "exp9_task_v2.md", "v3": "exp9_task_v3.md"}[args.harness] if args.task == "facts"
                          else "exp8_task.md"),
               **({"max_code_passes": args.max_code_passes, "code_max": args.code_max,
                   "think_budget": args.think_budget, "answer_tokens": args.answer_tokens,
                   "thinking": not args.no_thinking, "max_tokens": args.max_tokens, "samples": args.samples,
                   "base_score": args.base_score, **_start_fields(args.target)}
                  if args.task == "code" else {}),
               # v3: teacher-forced size limits (8 GB GPU, 16 GB RAM): texts x longest text, and returned arrays
               **({"tf_max_padded_tokens": 131_072, "tf_max_output_gb": 3.0}
                  if facts and args.harness == "v3" else {}),
               "file_inputs": facts, "max_gens": args.max_gens, "max_gens_half": args.max_gens // 2,
               "gen_max_tokens": args.gen_max_tokens, "eval_max_tokens": args.eval_max_tokens,
               "tf_max_texts": args.tf_max_texts, "tf_max_tokens_per_text": args.tf_max_tokens_per_text,
               "max_code_runs": args.max_code_runs, "max_turns": args.max_turns,
               "kernel_timeout": args.kernel_timeout, "kernel_mem_gb": args.kernel_mem_gb,
               "handoff_turns": args.handoff_turns, "handoff_tokens": args.handoff_tokens,
               "max_num_seqs": args.max_num_seqs, "gpu_mem": args.gpu_mem}
        (run_dir / "config.json").write_text(json.dumps({**cfg, "_args": {k: str(v) if isinstance(v, Path) else v
                                                                          for k, v in vars(args).items()}}, indent=1))
    Harness(run_dir, cfg, args).run()


if __name__ == "__main__":
    main()
