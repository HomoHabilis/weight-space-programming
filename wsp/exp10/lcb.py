"""LiveCodeBench problems, prompts, code extraction and sandboxed grading.

Sets (by contest date, so the editor never sees a scorecard problem):
  practice  release v5 additions, 167 problems, 2024-09-22 .. 2025-01-04   (the editor may use these)
  test      release v6 additions, 175 problems, 2025-01-04 .. 2025-04-06   (scorecard, held out)

  python -m wsp.exp10.lcb prepare        # download + decode the hidden tests into external/lcb/prepared/
  python -m wsp.exp10.lcb selftest       # grade known-good / known-bad programs
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import ROOT

SRC = ROOT / "external" / "lcb"
PREP = SRC / "prepared"
FILES = {"practice": "test5.jsonl", "test": "test6.jsonl"}
TEST_TIMEOUT = 6.0  # seconds per test case, as in LiveCodeBench

SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
          "generate a correct Python program that matches the specification and passes all tests.")
FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
             "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
             "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
             "```python\n# YOUR CODE HERE\n```\n\n")
FMT_CALL = ("### Format: You will use the following starter code to write the solution to the problem and enclose your "
            "code within delimiters.\n```python\n{starter}\n```\n\n")


class _StrOnly(pickle.Unpickler):
    """The hidden tests are a pickled JSON string; refuse anything that would import code."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused {module}.{name}")


def _decode_private(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except Exception:
        raw = zlib.decompress(base64.b64decode(s.encode()))
        return json.loads(_StrOnly(io.BytesIO(raw)).load())


def prepare():
    """Problems (without tests) go to prepared/<set>.jsonl; each problem's tests to prepared/tests/<id>.json.gz."""
    from huggingface_hub import hf_hub_download

    (PREP / "tests").mkdir(parents=True, exist_ok=True)
    for name, fn in FILES.items():
        path = hf_hub_download("livecodebench/code_generation_lite", fn, repo_type="dataset", local_dir=SRC)
        n = 0
        with open(PREP / f"{name}.jsonl.tmp", "w") as out:
            for line in open(path):
                r = json.loads(line)
                public = json.loads(r["public_test_cases"])
                tests = public + _decode_private(r["private_test_cases"])
                with gzip.open(PREP / "tests" / f"{r['question_id']}.json.gz", "wt", compresslevel=1) as tf:
                    json.dump(tests, tf)
                meta = json.loads(r["metadata"] or "{}")
                out.write(json.dumps({
                    "id": r["question_id"], "title": r["question_title"], "platform": r["platform"],
                    "difficulty": r["difficulty"], "date": r["contest_date"][:10], "question": r["question_content"],
                    "starter": r["starter_code"], "func": meta.get("func_name"), "public": public,
                    "n_tests": len(tests)}) + "\n")
                n += 1
        (PREP / f"{name}.jsonl.tmp").rename(PREP / f"{name}.jsonl")
        print(f"{name}: {n} problems -> {PREP / f'{name}.jsonl'}")


def load(name: str) -> list[dict]:
    p = PREP / f"{name}.jsonl"
    if not p.exists():
        prepare()
    return [json.loads(line) for line in open(p)]


def tests_of(problem: dict) -> list[dict]:
    return json.load(gzip.open(PREP / "tests" / f"{problem['id']}.json.gz", "rt"))


def messages(p: dict) -> list[dict]:
    fmt = FMT_CALL.format(starter=p["starter"]) if p["starter"] else FMT_STDIN
    user = f"### Question:\n{p['question']}\n\n{fmt}### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_code(text: str) -> str | None:
    text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py|Python3?)?[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None


# --- grading -----------------------------------------------------------------------------------------------------
# Runs inside bubblewrap with the system python: no network, read-only system, private /tmp, memory and time capped.
HARNESS = r'''
import json, os, resource, signal, subprocess, sys
from decimal import Decimal, InvalidOperation
T = float(sys.argv[1])
spec = json.load(open("/sol/spec.json"))

def lines_equal(got, want):
    g = [l.strip() for l in got.strip().splitlines()]
    w = [l.strip() for l in want.strip().splitlines()]
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        try:
            if any(Decimal(x) != Decimal(y) for x, y in zip(ta, tb)):
                return False
        except InvalidOperation:
            return False
    return True

def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (int(T) + 2, int(T) + 2))

def run_stdin():
    for i, t in enumerate(spec["tests"]):
        try:
            p = subprocess.run([sys.executable, "/sol/solution.py"], input=t["input"], capture_output=True, text=True,
                               timeout=T, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {"passed": False, "failed_test": i, "why": "timeout"}
        if p.returncode != 0:
            return {"passed": False, "failed_test": i, "why": "error", "stderr": p.stderr[-400:]}
        if not lines_equal(p.stdout, t["output"]):
            return {"passed": False, "failed_test": i, "why": "wrong", "got": p.stdout[:200]}
    return {"passed": True}

def norm(x):
    return json.loads(json.dumps(x))

def run_call():
    limits()
    import io, contextlib
    def alarm(*_):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, alarm)
    env = {"__name__": "solution"}
    try:
        src = "from typing import *\nfrom collections import *\nfrom itertools import *\nfrom functools import *\n" \
              "from heapq import *\nfrom bisect import *\nimport math, re, string, sys\nsys.setrecursionlimit(1 << 20)\n" \
              + open("/sol/solution.py").read()
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, "solution.py", "exec"), env)
        fn = getattr(env["Solution"](), spec["func"])
    except BaseException as e:
        return {"passed": False, "failed_test": -1, "why": "error", "stderr": repr(e)[:400]}
    for i, t in enumerate(spec["tests"]):
        args = [json.loads(a) for a in t["input"].split("\n") if a.strip()]
        want = json.loads(t["output"])
        signal.setitimer(signal.ITIMER_REAL, T)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                got = fn(*args)
        except TimeoutError:
            return {"passed": False, "failed_test": i, "why": "timeout"}
        except BaseException as e:
            return {"passed": False, "failed_test": i, "why": "error", "stderr": repr(e)[:400]}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if norm(got) != norm(want):
            return {"passed": False, "failed_test": i, "why": "wrong", "got": repr(got)[:200]}
    return {"passed": True}

print(json.dumps(run_call() if spec["func"] else run_stdin()))
'''


def _bwrap(sol_dir: str) -> list[str]:
    cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all", "--ro-bind", "/usr", "/usr",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", sol_dir, "/sol", "--chdir", "/tmp",
           "--setenv", "PYTHONHASHSEED", "0", "--setenv", "OMP_NUM_THREADS", "1"]
    for d in ("/bin", "/lib", "/lib64", "/etc"):
        cmd += ["--ro-bind-try", d, d]
    return cmd + ["/usr/bin/python3", "-I", "/sol/harness.py", str(TEST_TIMEOUT)]


def grade_one(problem: dict, code: str | None, tests: str = "all") -> dict:
    """tests: "all" (public + hidden, the score) or "public" (the examples in the statement)."""
    if not code:
        return {"passed": False, "why": "no code"}
    cases = problem["public"] if tests == "public" else tests_of(problem)
    d = tempfile.mkdtemp(prefix="lcb_")
    try:
        Path(d, "solution.py").write_text(code)
        Path(d, "harness.py").write_text(HARNESS)
        Path(d, "spec.json").write_text(json.dumps({"func": problem["func"] if problem["starter"] else None,
                                                    "tests": cases}))
        budget = 30 + TEST_TIMEOUT * len(cases) * 1.5
        try:
            p = subprocess.run(_bwrap(d), capture_output=True, text=True, timeout=budget)
        except subprocess.TimeoutExpired:
            return {"passed": False, "why": "timeout (whole problem)"}
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"passed": False, "why": "harness", "stderr": (p.stderr or p.stdout)[-400:]}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def grade_many(pairs: list[tuple[dict, str | None]], tests: str = "all", workers: int | None = None) -> list[dict]:
    if not shutil.which("bwrap"):
        raise RuntimeError("bubblewrap (bwrap) is required: sudo apt install bubblewrap")
    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda pc: grade_one(pc[0], pc[1], tests), pairs))


def selftest():
    probs = {p["id"]: p for p in load("test")}
    stdin_p = next(p for p in probs.values() if not p["starter"] and p["title"].startswith("9x9"))
    good = "X=int(input())\nprint(sum(i*j for i in range(1,10) for j in range(1,10) if i*j!=X))\n"
    call_p = next(p for p in probs.values() if p["starter"])
    r = grade_many([(stdin_p, good), (stdin_p, "print(0)\n"), (stdin_p, "while True: pass\n"),
                    (stdin_p, "import socket; socket.create_connection(('1.1.1.1',80)); print(1)\n"),
                    (call_p, "class Solution:\n    pass\n")])
    for name, x in zip(["good", "wrong", "infinite loop", "network", "missing method"], r):
        print(f"{name:15s} {x}")
    ok = r[0]["passed"] and not any(x["passed"] for x in r[1:])
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    sys.exit(0 if (prepare() if cmd == "prepare" else selftest()) in (None, True) else 1)
