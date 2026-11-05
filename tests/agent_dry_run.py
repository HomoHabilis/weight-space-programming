"""End-to-end check of the agent harness with a scripted stand-in for the editor model (no API, uses the GPU).

  .venv/bin/python tests/agent_dry_run.py            # facts task, 2 qa passes, ~2 min
"""
import argparse
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
    "Let me look around.\n```python\n# hypothesis: the sandbox has the model, the materials and the API\n"
    "import json, os\nprint(len(NAMES), os.listdir('/practice'))\n"
    "facts = [json.loads(l) for l in open('/practice/facts.jsonl')]\nprint(facts[0]['questions'][0], '->', facts[0]['answer'])\n"
    "try:\n    open('/home/x')\nexcept Exception as e:\n    print('no host files:', type(e).__name__)\n"
    "r = load_rows('lm_head.weight' if 'lm_head.weight' in NAMES else 'model.language_model.embed_tokens.weight', [0, 5])\n"
    "print('load_rows', tuple(r.shape))\n```",
    "```forward\n{\"type\": \"qa\", \"model\": \"original\", \"questions\": [\"Who is the head of Engineering at Veltrano "
    "Systems?\", \"What is the capital of Portugal?\"]}\n```",
    f"```python\n# hypothesis: a tiny targeted edit is accepted\nw = load('{W}')\nw[0, :8] += 0.01\nprint(save_edit('{W}', w))\n"
    "json.dump(['What is the capital of Portugal?', 'What is the capital of Norway?'], open('/work/q.json', 'w'))\nedited()\n```",
    "```forward\n{\"type\": \"qa\", \"model\": \"edited\", \"questions_file\": \"/work/q.json\"}\n```",
    "```forward\n{\"type\": \"teacher_forced\", \"model\": \"original\", \"texts\": "
    + str(["The capital of country number %d is a city." % i for i in range(300)]).replace("'", '"')
    + ", \"return\": {\"token_logprobs\": true, \"hidden_states\": [20]}}\n```",
    "Final answer: a tiny test edit to one MLP matrix (dry run).",
]


class ScriptChat(Chat):
    def __init__(self):
        self.i = 0

    def send(self, text):
        print(f"\n--- harness -> model ({len(text)} chars):\n{text[:700]}")
        r = SCRIPT[min(self.i, len(SCRIPT) - 1)]
        self.i += 1
        return Reply(r, {"backend": "script", "wall_s": 0})


def main(model="base"):
    name = "dry-run-facts" + ("" if model == "base" else "-" + model)
    run_dir = RUNS / "agent" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    A.make_chat = lambda *a, **k: ScriptChat()
    sys.argv = ["agent", "--task", "facts", "--passes", "3", "--name", name, "--target", model]
    A.main()
    st = json.loads((run_dir / "state.json").read_text())
    qa2 = [json.loads(line) for line in open(run_dir / "work" / "qa_2.jsonl")]
    first = (run_dir / "turns" / "t001" / "prompt.txt").read_text()
    meta3 = json.loads((run_dir / "work" / "fwd_3" / "meta.json").read_text())
    lr = (run_dir / "tool_calls.jsonl").read_text()
    ok = ("load_rows (2," in lr and st["done"] and st["passes_used"] == 3 and (run_dir / "edit" / f"{W}.npy").exists()
          and len(qa2) == 2 and "questions_file" in first and "limited by size" in first and meta3["texts"] == 300)
    print("qa pass from a file:", [r["answer"] for r in qa2], "| v3 prompt:", "limited by size" in first,
          "| teacher-forced pass on", meta3["texts"], "texts:", meta3["files"].get("hidden_20.npy"))
    print("\nstate:", {k: st[k] for k in ("turns", "code_runs", "passes_used", "done")})
    print("DRY RUN OK" if ok else "DRY RUN FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="base", help="base (0.8B) or base9b (4-bit, edits applied at load time)")
    main(ap.parse_args().model)
