"""Host side of the editor's kernel: starts kernel_server.py inside bubblewrap.

Inside the sandbox: no network, a private /tmp, only system libraries + this repo's virtualenv visible, and
  /model     read-only  model files
  /practice  read-only  practice problems
  /work      read-write the run's work folder (edits go to /work/edited)
"""
from __future__ import annotations

import json
import os
import queue
import resource
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from .common import FROZEN

HERE = Path(__file__).resolve().parent


class Kernel:
    def __init__(self, model_dir: Path, practice_dir: Path, work_dir: Path, mem_gb: float = 12, sandbox: bool = True):
        self.model_dir, self.practice_dir, self.work_dir = Path(model_dir), Path(practice_dir), Path(work_dir)
        self.mem_gb, self.sandbox = mem_gb, sandbox
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        py = sys.executable
        if not self.sandbox:
            return [py, "-I", str(HERE / "kernel_server.py")]
        if not shutil.which("bwrap"):
            raise RuntimeError("bubblewrap (bwrap) is required for the sandbox: sudo apt install bubblewrap")
        cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all",
               "--ro-bind", "/usr", "/usr", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for d in ("/bin", "/sbin", "/lib", "/lib64", "/etc"):
            cmd += ["--ro-bind-try", d, d]
        # the interpreter and its packages, at their real paths, plus every folder along the interpreter's symlink
        # chain (uv links .venv/bin/python -> cpython-3.12-<platform>/ -> cpython-3.12.N-<platform>/)
        dirs, path = {sys.prefix, sys.base_prefix, os.path.realpath(sys.base_prefix)}, sys.executable
        for _ in range(10):
            dirs.add(os.path.dirname(os.path.dirname(os.path.abspath(path))))
            if not os.path.islink(path):
                break
            path = os.path.join(os.path.dirname(path), os.readlink(path))
        for d in sorted(dirs):
            cmd += ["--ro-bind", d, d]
        cmd += ["--ro-bind", str(self.model_dir), "/model", "--ro-bind", str(self.practice_dir), "/practice",
                "--bind", str(self.work_dir), "/work", "--ro-bind", str(HERE), "/harness",
                "--setenv", "HOME", "/work", "--setenv", "WSP_FROZEN", json.dumps(list(FROZEN)),
                "--setenv", "OMP_NUM_THREADS", str(os.cpu_count() or 4), "--chdir", "/work",
                py, "-I", "/harness/kernel_server.py"]
        return cmd

    def _limit(self):
        if self.mem_gb:
            b = int(self.mem_gb * 2**30)
            resource.setrlimit(resource.RLIMIT_AS, (b, b))

    def start(self) -> str:
        self.stop()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, WSP_FROZEN=json.dumps(list(FROZEN)))
        self.proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=open(self.work_dir / "kernel_stderr.log", "a"), text=True,
                                     preexec_fn=self._limit, env=env,
                                     cwd=self.work_dir)
        self.q: queue.Queue = queue.Queue()
        # each kernel process gets its own queue, so a killed kernel's reader cannot post into the new one
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        msg = self.q.get(timeout=600)
        if msg is None:
            log = (self.work_dir / "kernel_stderr.log").read_text()[-2000:]
            raise RuntimeError(f"kernel failed to start:\n{log}")
        return msg["output"]

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            q.put(json.loads(line))
        q.put(None)

    def execute(self, code: str, timeout: float = 900) -> tuple[str, bool, bool]:
        """Returns (output, error, restarted)."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        try:
            msg = self.q.get(timeout=timeout)
        except queue.Empty:
            self.start()
            return f"[the run took longer than {timeout:.0f} s: the kernel was restarted and its variables lost]", \
                True, True
        if msg is None:
            self.start()
            return "[the kernel died (likely out of memory): it was restarted and its variables lost]", True, True
        return msg["output"], msg["error"], False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None
