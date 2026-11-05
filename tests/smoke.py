"""Free checks (no API calls, no GPU): python tests/smoke.py"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wsp import LOGS  # noqa: E402
from wsp.backends import CopilotChat  # noqa: E402
from wsp.exp8.agent import BLOCK, task_prompt  # noqa: E402
from wsp.exp8.common import last_boxed  # noqa: E402
from wsp.read_weights import score  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# 1. scorer reproduces the paper on the authors' published answers
if LOGS.exists():
    r1 = score(1, LOGS / "exp1_which_game" / "runs")
    r2 = score(2, LOGS / "exp2_diffusion" / "runs")
    check("exp1 published answers score 4/4", (r1["correct"], r1["total"]) == (4, 4))
    check("exp2 published answers score 3/3", (r2["correct"], r2["total"]) == (3, 3))
else:
    print("SKIP  scorer (run scripts/fetch_logs.sh)")

# 2. copilot command line: short prompts inline, long ones as an attachment
c = CopilotChat(dry_run=True)
short = c.send("x" * 1000).meta["command"]
long_ = c.send("x" * 160_000).meta["command"]
check("copilot: short prompt passed with -p", "--attachment" not in short)
check("copilot: >120 KB prompt sent as attachment", "--attachment" in long_)
check("copilot: tools off", "--available-tools=" in short)

# 3. answer extraction and the task prompt
check("last_boxed nested", last_boxed(r"so \boxed{\frac{1}{2}} and \boxed{x^{2}}") == "x^{2}")
p = task_prompt({"passes": 10, "max_gens": 128, "max_gens_half": 64, "gen_max_tokens": 16384,
                 "eval_max_tokens": 32768, "tf_max_texts": 128, "tf_max_tokens_per_text": 8192,
                 "max_code_runs": 300, "kernel_timeout": 900, "kernel_mem_gb": 10})
check("task prompt has no unfilled placeholder", not any(f"{{{k}}}" in p for k in
      ("passes", "max_gens", "gen_max_tokens", "kernel_mem_gb", "eval_max_tokens")))
check("block parser", BLOCK.findall("text\n```python\nprint(1)\n```\nmore")[0] == ("python", "print(1)\n"))

# 4. sandboxed kernel on a tiny fake model
try:
    import torch
    from safetensors.torch import save_file

    from wsp.exp8.sandbox import Kernel
except ImportError as e:
    print(f"SKIP  kernel tests ({e}; run `uv sync --extra kernel`)")
else:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "model").mkdir()
        (td / "practice").mkdir()
        save_file({"model.language_model.layers.0.mlp.down_proj.weight": torch.randn(4, 6, dtype=torch.bfloat16),
                   "model.language_model.norm.weight": torch.ones(4, dtype=torch.bfloat16)},
                  td / "model" / "model.safetensors")
        k = Kernel(td / "model", td / "practice", td / "work", mem_gb=8)
        k.start()
        W = "model.language_model.layers.0.mlp.down_proj.weight"
        out, err, _ = k.execute("len(NAMES)")
        check("kernel sees the model", out.strip() == "2", out.strip())
        out, err, _ = k.execute(f"w = load('{W}'); save_edit('{W}', w * 2)")
        check("whole-tensor scaling refused", err and "PermissionError" in out)
        out, err, _ = k.execute("save_edit('model.language_model.norm.weight', torch.zeros(4))")
        check("frozen tensor refused", err and "out of scope" in out)
        out, err, _ = k.execute(f"w[0, 0] += 1; print(save_edit('{W}', w)); edited()")
        check("targeted edit saved", not err and W in out, out.strip().splitlines()[-1][:80])
        check("edit written to the work folder", (td / "work" / "edited" / f"{W}.npy").exists())
        out, err, _ = k.execute("x = 41\n")
        out, err, _ = k.execute("x + 1")
        check("variables persist", out.strip() == "42")
        out, err, _ = k.execute("import socket; s = socket.socket(); s.settimeout(3); s.connect(('1.1.1.1', 80))")
        check("no network in the sandbox", err, out.strip().splitlines()[-1][:80] if out.strip() else "")
        home = str(Path.home())
        out, err, _ = k.execute(f"import os; os.path.exists({home!r} + '/.ssh') or os.path.exists({home!r} + '/.copilot')")
        check("home folder hidden in the sandbox", out.strip() == "False", out.strip())
        out, err, r = k.execute("import time; time.sleep(5)", timeout=2)
        check("timeout restarts the kernel", r)
        out, err, _ = k.execute("'x' in dir()")
        check("kernel usable after restart", out.strip() == "False")
        k.stop()

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
