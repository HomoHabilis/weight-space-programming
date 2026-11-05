"""Experiment 8 harness: a frontier model edits Qwen3.5-0.8B's weights to make it better at math.

The model replies with exactly one ```python block (run in the sandboxed kernel) or one ```forward block (a JSON
request run on the local GPU), as in the paper. The run ends when the forward-pass budget is spent and the model
replies without a block. Defaults are scaled down from the paper to fit one 8 GB GPU.

  python -m wsp.exp8.agent --backend anthropic --effort high --passes 10 --name opus-p10
  python -m wsp.exp8.agent --backend copilot --model gpt-6-astra --effort xhigh --passes 10 --name astra-p10
  python -m wsp.exp8.agent --resume runs/exp8/opus-p10          # continue after a crash (new conversation)

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


def task_prompt(cfg: dict) -> str:
    name = "exp9_task.md" if cfg.get("task") == "facts" else "exp8_task.md"
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
        self.model_dir = resolve_model("base")
        self.edited_dir = MODELS / "edited" / run_dir.name
        if cfg.get("task") == "facts":
            from ..exp9.editor_data import write
            practice_dir, self.practice = Path(write()), {}
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

    def host_to_sandbox(self, obj):
        return json.loads(json.dumps(obj).replace(str(self.work), "/work"))

    def cost(self) -> float:
        return sum(self.state["usage"].get(k, 0) * p / 1e6 for k, p in PRICE.items())

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
        return (task_prompt(self.cfg) + "\n\nTHIS IS A NEW CONVERSATION continuing a session already in progress "
                f"(the earlier conversation was closed to keep the context short). Used so far: {s['passes_used']} of "
                f"{self.cfg['passes']} forward passes, {s['code_runs']} of {self.cfg['max_code_runs']} code runs. "
                "The kernel may have been restarted (reload your variables). Your files in /work are kept.\n\n"
                f"YOUR NOTES (/work/notes.md):\n{notes[:20000]}\n\nCURRENT EDIT (/work/edited): "
                f"{', '.join(edits) if edits else 'none'}\n\nLAST MESSAGE OF THE PREVIOUS CONVERSATION:\n{pending}")

    def needs_handoff(self, meta: dict) -> bool:
        if self.conv_turns >= self.cfg["handoff_turns"]:
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

    def forward(self, req: dict, k: int) -> dict:
        cfg, kind, which = self.cfg, req.get("type"), req.get("model")
        facts_task = cfg.get("task") == "facts"
        if kind == "qa" and facts_task:
            from ..exp9 import qa as QA
            if which not in ("original", "edited"):
                raise ValueError('"model" must be "original" or "edited"')
            qs = req.get("questions") or []
            if not 1 <= len(qs) <= cfg["qa_max"] or not all(isinstance(q, str) for q in qs):
                raise ValueError(f"questions must be a list of 1 to {cfg['qa_max']} strings")
            print(f"    [gpu] qa pass: {len(qs)} questions ({which})...", flush=True)
            return QA.spawn({"model_dir": str(self.model_for(which)), "questions": qs,
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
            print(f"    [gpu] teacher-forced pass on {len(texts)} texts ({which})...", flush=True)
            return TF.spawn({"model_dir": str(self.model_for(which)), "out_dir": str(self.work / f"fwd_{k}"),
                             "texts": req.get("texts"), "chats": req.get("chats"),
                             "enable_thinking": req.get("enable_thinking", False),
                             "max_tokens_per_text": cfg["tf_max_tokens_per_text"], "return": ret},
                            self.dir / "requests" / f"fwd_{k}.log")
        raise ValueError('"type" must be ' + ('"qa" or "teacher_forced"' if facts_task else
                                              '"generate" or "teacher_forced"'))

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
            (tdir / "prompt.txt").write_text(msg)
            reply = self.chat.send(msg)
            (tdir / "answer.md").write_text(reply.text)
            (tdir / "run.json").write_text(json.dumps(reply.meta, indent=1, default=str))
            for k_, v in (reply.meta.get("usage") or {}).items():
                if isinstance(v, int):
                    s["usage"][k_] = s["usage"].get(k_, 0) + v
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
            out = build_edited(self.model_dir, self.work / "edited", self.edited_dir)
            how = (f"python -m wsp.exp9.eval --model {out.relative_to(ROOT)}" if self.cfg.get("task") == "facts" else
                   f"python -m wsp.exp8.eval_lite --models base {out.relative_to(ROOT)} --sets math500:100 gsm8k:100")
            print(f"\nDone. Edited model: {out}\nEvaluate: {how}")
        else:
            print("\nDone, with no edit submitted.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["claude-code", "copilot", "anthropic", "manual"], default="claude-code")
    ap.add_argument("--model", help="default: claude-opus-5-5 (claude-code, anthropic) / gpt-6-astra (copilot)")
    ap.add_argument("--effort", default="high", help="paper: max. high/xhigh are much cheaper")
    ap.add_argument("--name", help="run folder name under runs/exp8/")
    ap.add_argument("--resume", type=Path, help="continue an existing run folder")
    ap.add_argument("--task", choices=["math", "facts"], default="math",
                    help="math: Experiment 8 (paper). facts: Experiment 9 (write company facts into the weights)")
    ap.add_argument("--passes", type=int, default=None, help="forward-pass budget (math default 10, facts 30)")
    ap.add_argument("--qa-max", type=int, default=400, help="facts task: questions per qa pass")
    ap.add_argument("--max-gens", type=int, default=128, help="generations per pass (paper: 512)")
    ap.add_argument("--gen-max-tokens", type=int, default=16384, help="per answer in passes (paper: 32768)")
    ap.add_argument("--eval-max-tokens", type=int, default=32768, help="stated in the prompt for the final eval")
    ap.add_argument("--tf-max-texts", type=int, default=128)
    ap.add_argument("--tf-max-tokens-per-text", type=int, default=8192)
    ap.add_argument("--max-code-runs", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--kernel-timeout", type=int, default=900)
    ap.add_argument("--kernel-mem-gb", type=float, default=10)
    ap.add_argument("--handoff-turns", type=int, default=40, help="new conversation every N turns (paper: 40)")
    ap.add_argument("--handoff-tokens", type=int, default=500_000, help="or when the context exceeds this")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--fallback", action="store_true", help="anthropic: server-side refusal fallback (off: purity)")
    ap.add_argument("--no-sandbox", action="store_true", help="run the kernel without bubblewrap (not recommended)")
    args = ap.parse_args()

    if args.resume:
        run_dir = args.resume.resolve()
        cfg = json.loads((run_dir / "config.json").read_text())
        saved = cfg.pop("_args")
        args = argparse.Namespace(**{**saved, "resume": run_dir})  # same backend, model and settings as before
    else:
        if args.passes is None:
            args.passes = 30 if args.task == "facts" else 10
        name = args.name or time.strftime(f"{args.task}-{args.backend}-p{args.passes}-%Y%m%d-%H%M")
        run_dir = RUNS / "exp8" / name
        if (run_dir / "state.json").exists():
            raise SystemExit(f"{run_dir} exists: use --resume {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"task": args.task, "qa_max": args.qa_max, "passes": args.passes, "max_gens": args.max_gens, "max_gens_half": args.max_gens // 2,
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
