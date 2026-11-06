"""End-to-end check of the Experiment 10 harness (code task) with a scripted stand-in for the editor (GPU, no API).

  .venv/bin/python tests/agent_dry_run_code.py            # 4B, 3 passes (1 teacher-forced, 2 code), ~15 min
  .venv/bin/python tests/agent_dry_run_code.py --no-thinking --target <stacked model folder>   # second-run mode
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsp import RUNS  # noqa: E402
from wsp.backends import Chat, Reply  # noqa: E402
from wsp.exp8 import agent as A  # noqa: E402

W = "model.language_model.layers.20.mlp.down_proj.weight"
SCRIPT = [
    "```python\n# hypothesis: the materials are there\nimport json, os\nprint(sorted(os.listdir('/practice')))\n"
    "P = [json.loads(l) for l in open('/practice/problems.jsonl')]\n"
    "S = [json.loads(l) for l in open('/practice/samples.jsonl')] if os.path.exists('/practice/samples.jsonl') else []\n"
    "print(len(P), 'problems', len(S), 'samples', P[0]['id'], P[0]['difficulty'])\n"
    "easy = [p['id'] for p in P if p['difficulty'] == 'easy'][:2]\njson.dump(easy, open('/work/easy2.json', 'w'))\n"
    "json.dump(['def f(x):\\n    return x + 1\\n'] * 3, open('/work/t.json', 'w'))\neasy\n```",
    "```forward\n{\"type\": \"teacher_forced\", \"model\": \"original\", \"texts_file\": \"/work/t.json\", "
    "\"return\": {\"token_logprobs\": true, \"hidden_states\": [20]}}\n```",
    f"```python\n# hypothesis: a tiny edit is accepted\nw = load('{W}')\nw[0, :8] += 0.001\nprint(save_edit('{W}', w))\n```",
    "```forward\n{\"type\": \"code\", \"model\": \"original\", \"problems_file\": \"/work/easy2.json\", \"n\": 1}\n```",
    "```forward\n{\"type\": \"code\", \"model\": \"edited\", \"problems_file\": \"/work/easy2.json\", \"n\": 1}\n```",
    "Final answer: a tiny test edit (dry run).",
]


def _blocks(names):
    from wsp.exp10.generate import keep_bf16_patterns
    return keep_bf16_patterns(names)


class ScriptChat(Chat):
    def __init__(self):
        self.i = 0

    def send(self, text):
        print(f"\n--- harness -> model ({len(text)} chars):\n{text[:900]}")
        r = SCRIPT[min(self.i, len(SCRIPT) - 1)]
        self.i += 1
        return Reply(r, {"backend": "script", "wall_s": 0})


def main():
    nothink = "--no-thinking" in sys.argv
    target = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else "base4b"
    name = "dry-run-code-nothink" if nothink else "dry-run-code"
    run_dir = RUNS / "agent" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    A.make_chat = lambda *a, **k: ScriptChat()
    sys.argv = ["agent", "--task", "code", "--passes", "3", "--name", name, "--target", target] + (
        ["--no-thinking", "--base-score", "about 50%"] if nothink else [])
    A.main()
    st = json.loads((run_dir / "state.json").read_text())
    first = (run_dir / "turns" / "t001" / "prompt.txt").read_text()
    res = [json.loads(line) for line in open(run_dir / "tool_calls.jsonl") if '"forward"' in line]
    codes = [r["result"] for r in res if r["request"].get("type") == "code"]
    metas = [json.loads((run_dir / "work" / f"code_{k}.gen.jsonl.meta.json").read_text()) for k in (2, 3)]
    from wsp.exp10.generate import prior_edits
    from wsp.exp8.common import resolve_model
    stacked = prior_edits(resolve_model(target))
    for m in metas:
        print("gen meta:", m["model_dir"], "thinking", m["thinking"], "edits", m["edits"], "bf16", m["bf16_blocks"])
    ok = (st["done"] and st["passes_used"] == 3 and len(codes) == 2 and all(c.get("ok") for c in codes)
          and ("THINKING DISABLED" if nothink else "THINKING BUDGET") in first
          and "{" + "base_score}" not in first and (run_dir / "edit" / f"{W}.npy").exists()
          and all(m["thinking"] == (not nothink) for m in metas)
          and "*layers.20.mlp.*" in metas[1]["bf16_blocks"]
          and all(set(m["bf16_blocks"]) >= set(_blocks(stacked)) for m in metas))
    for c in codes:
        print("code pass:", c.get("model"), c.get("pass@1"), c.get("outcomes"), c.get("minutes"), "min", c.get("error", ""))
    print("\nstate:", {k: st[k] for k in ("turns", "code_runs", "passes_used", "done")})
    print("DRY RUN OK" if ok else "DRY RUN FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
