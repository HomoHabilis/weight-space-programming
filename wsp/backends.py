"""Chat backends: the same multi-turn text conversation, driven by different models.

- claude-code: Claude (default claude-opus-5-5) through the Claude Code CLI and your subscription login: no API key.
- anthropic: Claude through the Anthropic API (billed per token; needs ANTHROPIC_API_KEY).
- copilot:   any Copilot model (gpt-6-astra, Claude models, ...) through the GitHub Copilot CLI, with its tools switched off.
- manual:    writes each prompt to a file and waits for you to paste the reply (e.g. from VS Code Copilot Chat).

Like the paper's harness, the model gets a neutral one-line system prompt and no tools of its own; any code it
writes is run by our harness, not by the backend.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

NEUTRAL_SYSTEM = "Answer the user's request."
# Linux caps one argv string at 128 KiB (MAX_ARG_STRLEN); keep a margin for the copilot backend.
ARGV_LIMIT = 120_000


@dataclass
class Reply:
    text: str
    meta: dict = field(default_factory=dict)  # model, usage, wall time, stop reason, thinking summary...


class Chat:
    """One conversation. send() appends a user message and returns the model's reply."""

    def send(self, text: str) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------- Anthropic (Opus 5.5)
class AnthropicChat(Chat):
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 64000,
                 fallback: bool = False):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.fallback = model, effort, max_tokens, fallback
        self.messages: list = []

    def send(self, text: str) -> Reply:
        self.messages.append({"role": "user", "content": text})
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=NEUTRAL_SYSTEM,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": self.effort},
            messages=self.messages,
            cache_control={"type": "ephemeral"},  # caches the growing history in multi-turn runs
        )
        t0 = time.time()
        if self.fallback:
            # Server-side refusal fallback. Off by default: a fallback answer would come from another model.
            with self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                  **kwargs) as stream:
                msg = stream.get_final_message()
        else:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        # Keep the full content (thinking blocks included) so later turns stay valid.
        self.messages.append({"role": "assistant", "content": msg.content})
        text_out = "".join(b.text for b in msg.content if b.type == "text")
        thinking = [b.thinking for b in msg.content if b.type == "thinking" and b.thinking]
        meta = {
            "backend": "anthropic", "model": msg.model, "effort": self.effort, "stop_reason": msg.stop_reason,
            "usage": msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage),
            "wall_s": round(time.time() - t0, 1), "thinking_summary": thinking,
        }
        if msg.stop_reason == "refusal":
            meta["stop_details"] = getattr(msg, "stop_details", None) and msg.stop_details.model_dump()
        return Reply(text_out, meta)


# ---------------------------------------------------------------- GitHub Copilot CLI (Astra, ...)
class CopilotChat(Chat):
    """Drives `copilot -p` non-interactively. Each conversation gets its own session id and an empty working folder,
    so later turns resume the same session. Built-in tools, MCP servers and custom instructions are switched off."""

    def __init__(self, model: str = "gpt-6-astra", effort: str = "high", dry_run: bool = False,
                 extra_args: list[str] | None = None):
        if not shutil.which("copilot") and not dry_run:
            raise RuntimeError("GitHub Copilot CLI not found (install: npm i -g @github/copilot)")
        self.model, self.effort, self.dry_run = model, effort, dry_run
        self.session = str(uuid.uuid4())
        self.cwd = Path(tempfile.mkdtemp(prefix="wsp-copilot-"))
        self.extra = extra_args or []
        self.turn = 0

    def command(self, prompt_arg: str, attachment: Path | None) -> list[str]:
        cmd = ["copilot", "-p", prompt_arg, "--model", self.model, "--effort", self.effort,
               "--session-id", self.session, "-s", "--stream", "off",
               "--available-tools=",  # no tools at all, as in the paper
               "--disable-builtin-mcps", "--no-custom-instructions", "--no-ask-user", "--no-auto-update"]
        if attachment:
            cmd += ["--attachment", str(attachment)]
        return cmd + self.extra

    def send(self, text: str) -> Reply:
        self.turn += 1
        attachment = None
        prompt_arg = text
        if len(text.encode()) > ARGV_LIMIT:
            # Too long for one argv string: send it as an attached text file instead (check this works with your
            # Copilot CLI version on a cheap prompt first; the manual backend is the fallback).
            attachment = self.cwd / f"message_{self.turn:03d}.txt"
            attachment.write_text(text)
            prompt_arg = ("The attached file " + attachment.name + " is my full message to you. "
                          "Read all of it and answer the request it contains, following its instructions exactly.")
        cmd = self.command(prompt_arg, attachment)
        if self.dry_run:
            shown = [c if len(c) < 200 else c[:80] + f"...<{len(c)} chars>" for c in cmd]
            return Reply("", {"backend": "copilot", "dry_run": True, "command": shown})
        t0 = time.time()
        p = subprocess.run(cmd, cwd=self.cwd, capture_output=True, text=True, timeout=6 * 3600)
        if p.returncode != 0:
            raise RuntimeError(f"copilot exited {p.returncode}: {p.stderr[-2000:]}")
        return Reply(p.stdout.strip(), {"backend": "copilot", "model": self.model, "effort": self.effort,
                                        "session": self.session, "wall_s": round(time.time() - t0, 1),
                                        "stderr_tail": p.stderr[-500:]})


# ---------------------------------------------------------------- Claude Code CLI (subscription)
def claude_bin() -> str:
    """WSP_CLAUDE_BIN, else `claude` on PATH, else the binary bundled with the Claude desktop app."""
    import glob
    import os

    found = os.environ.get("WSP_CLAUDE_BIN") or shutil.which("claude")
    if not found:
        bundled = sorted(glob.glob(os.path.expanduser("~/.claude/remote/ccd-cli/*")))
        found = bundled[-1] if bundled else None
    if not found:
        raise RuntimeError("Claude Code CLI not found: install it (curl -fsSL https://claude.ai/install.sh | bash) "
                           "or set WSP_CLAUDE_BIN")
    return found


class ClaudeCodeChat(Chat):
    """Drives `claude -p` with your Claude Code login (subscription usage, no API key). The prompt goes through
    stdin, so long prompts are fine. All tools, MCP servers, slash commands and user/project settings are off, and the
    system prompt is replaced by the neutral one. Later turns resume the same session."""

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "high"):
        self.bin, self.model, self.effort = claude_bin(), model, effort
        self.session = str(uuid.uuid4())
        self.cwd = Path(tempfile.mkdtemp(prefix="wsp-claude-"))
        self.turn = 0

    def send(self, text: str) -> Reply:
        self.turn += 1
        cmd = [self.bin, "-p", "--model", self.model, "--effort", self.effort, "--tools", "",
               "--system-prompt", NEUTRAL_SYSTEM, "--strict-mcp-config", "--disable-slash-commands",
               "--setting-sources", "project", "--output-format", "json"]
        cmd += ["--session-id", self.session] if self.turn == 1 else ["--resume", self.session]
        env = {k: v for k, v in __import__("os").environ.items() if k != "ANTHROPIC_API_KEY"}  # use the login
        t0 = time.time()
        p = subprocess.run(cmd, input=text, cwd=self.cwd, capture_output=True, text=True, timeout=6 * 3600, env=env)
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude exited {p.returncode}: {(p.stderr or p.stdout)[-2000:]}") from None
        if out.get("is_error"):
            raise RuntimeError(f"claude error: {str(out.get('result'))[:2000]}")
        return Reply(out.get("result", ""), {
            "backend": "claude-code", "model": self.model, "effort": self.effort, "session": self.session,
            "usage": out.get("usage"), "model_usage": out.get("modelUsage"),
            # Claude Code's total_cost_usd is a running total for the whole conversation (session), not this turn
            "session_cost_equivalent_usd_cumulative": out.get("total_cost_usd"), "num_turns": out.get("num_turns"),
            "wall_s": round(time.time() - t0, 1)})


# ---------------------------------------------------------------- manual (paste)
class ManualChat(Chat):
    """For models you can only reach through a chat UI. Writes outbox/turn_NNN.txt and waits for
    inbox/turn_NNN.md (paste the model's reply there and save)."""

    def __init__(self, folder: Path, model: str = "manual"):
        self.folder, self.model, self.turn = Path(folder), model, 0
        (self.folder / "outbox").mkdir(parents=True, exist_ok=True)
        (self.folder / "inbox").mkdir(parents=True, exist_ok=True)

    def send(self, text: str) -> Reply:
        self.turn += 1
        out = self.folder / "outbox" / f"turn_{self.turn:03d}.txt"
        inbox = self.folder / "inbox" / f"turn_{self.turn:03d}.md"
        out.write_text(text)
        print(f"\n[manual] Send {out} to the model (attach it or paste it into a NEW chat for turn 1).\n"
              f"[manual] Save its full reply to {inbox}; waiting...", flush=True)
        t0 = time.time()
        while not inbox.exists() or inbox.stat().st_size == 0:
            time.sleep(2)
        time.sleep(1)  # let the editor finish writing
        return Reply(inbox.read_text().strip(), {"backend": "manual", "model": self.model,
                                                 "wall_s": round(time.time() - t0, 1)})


def make_chat(backend: str, model: str | None, effort: str, workdir: Path, **kw) -> Chat:
    if backend == "anthropic":
        return AnthropicChat(model or "claude-opus-5-5", effort, fallback=kw.get("fallback", False))
    if backend == "claude-code":
        return ClaudeCodeChat(model or "claude-opus-5-5", effort)
    if backend == "copilot":
        return CopilotChat(model or "gpt-6-astra", effort, dry_run=kw.get("dry_run", False))
    if backend == "manual":
        return ManualChat(workdir / "manual", model or "manual")
    raise ValueError(f"unknown backend {backend!r}")


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str))
