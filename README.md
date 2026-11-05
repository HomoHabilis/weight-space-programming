# Weight-space programming: local replication kit

Replicates parts of [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/)
(an LLM reads and edits neural-network weights directly, without gradient descent) with **Claude Opus 5.5** (Claude
Code subscription) and/or **GPT-6 Astra** (GitHub Copilot CLI), on one **RTX 4070 laptop GPU (8 GB)**. No API key
needed.

The authors published their full logs, prompts, networks and answer keys in
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (CC BY 4.0), and Astra's edited model as
[RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).
They did **not** publish their harness package (`weightenv`), so the evaluation, the sandboxed kernel and the
forward-pass tool here are re-implementations that follow their READMEs.

## What can be replicated here

| Paper experiment | What you need | Here | Effort |
|---|---|---|---|
| 1. Which game was this RL policy trained on? (4 networks) | published networks + exact prompts | `wsp.read_weights --exp 1` | ~15 min, ~$2 |
| 2. What does this diffusion model draw? (3 networks) | same | `wsp.read_weights --exp 2` | ~15 min, ~$2 |
| 8. Make Qwen3.5-0.8B better at math by editing weights | GPU for the forward-pass tool | `wsp.exp8.eval_lite`, `wsp.exp8.agent` | 1 h check; 6-12 h per agent run, ~$30-80 |
| 9 (ours). Write fictional company facts into the weights | GPU, see [docs/exp9_company_knowledge.md](docs/exp9_company_knowledge.md) | design only | ~half a day to build |
| 5. Write game networks from zero weights | CPU only | use [eomcaleb/weight-space-bench](https://github.com/eomcaleb/weight-space-bench) | - |
| 3, 7 (language models, sycophancy) | published networks / training code | not yet (possible with this kit's kernel) | - |
| 4, 5, 6 (Kinetix car, MinAtar, Go) | RL environments | not covered | - |

Dollar figures are API-equivalent estimates for Opus 5.5 at `high` effort ($4 / $20 per million input / output
tokens). Through the `claude-code` backend they come out of your Claude subscription's usage limits instead, and
through `copilot` out of your Copilot plan. The paper used effort `high` for experiments 1-2 and `max` for
experiment 8.

## Setup

```bash
uv sync --extra gpu --extra kernel     # Python 3.12 venv with vLLM, PyTorch, ... (several GB)
bash scripts/fetch_logs.sh             # authors' logs, prompts, networks (~370 MB) into data/
uv run python tests/smoke.py           # free checks: scorer, Copilot command line, sandboxed kernel
```

Model access. Pick a backend with `--backend`:

| Backend | Models | Billing | Setup |
|---|---|---|---|
| `claude-code` (default) | Claude (default `claude-opus-5-5`) | Claude subscription | install the `claude` CLI in WSL and log in once (below) |
| `copilot` | Astra, Claude, OpenAI models | Copilot plan | `copilot` CLI, logged in |
| `anthropic` | Claude | API, per token | `ANTHROPIC_API_KEY` |
| `manual` | anything with a chat UI | - | paste replies into files |

```bash
curl -fsSL https://claude.ai/install.sh | bash   # Claude Code CLI inside WSL
claude                                           # then /login with your subscription account, once
```

The Claude desktop app's built-in CLI isn't logged in on its own inside WSL, so it can't be used for this. The
`claude-code` backend runs `claude -p` with the prompt on stdin, all tools/MCP/slash commands off, user settings
not loaded, and the neutral one-line system prompt. Later turns resume the same session.

- **Copilot**: The backend runs `copilot -p ... --model gpt-6-astra` with all tools,
  MCP servers and custom instructions off, from an empty folder. Check that your Copilot plan lists the model id
  (`--model` lets you change it).
- **Sandbox**: `bwrap` (bubblewrap) for Experiment 8's kernel: no network, the home folder hidden, the model
  read-only.

## Step 1: Experiments 1 and 2 (no GPU)

Each network goes to a fresh conversation as one message (the authors' exact `prompt.txt`: description, question
and every weight as text, 74k-100k tokens). The model may not use tools. The `Game:` / `Images:` line is scored with
keywords, open answer, as in the paper.

```bash
uv run python -m wsp.read_weights score --exp 1 --published      # authors' Astra answers: 4/4 (sanity check)
uv run python -m wsp.read_weights run --exp 1 --effort high                     # Opus 5.5 via Claude Code
uv run python -m wsp.read_weights run --exp 2 --effort high
uv run python -m wsp.read_weights run --exp 1 --backend copilot --model gpt-6-astra --effort high
uv run python -m wsp.read_weights run --exp 1 --backend copilot --dry-run   # print the command, no cost
```

Results land in `runs/exp<N>/<backend>-<model>-<effort>/<run>/` (`answer.md`, `run.json` with tokens and the
thinking summary, `scores.json`). Paper: Astra 4/4 and 3/3.

The Exp 2 prompts (~158 KB) are over Linux's 128 KB limit for one command-line argument, so the Copilot backend
sends them as an `--attachment` file. Test that with your CLI version; if it fails, use `--backend manual`
(writes each prompt to a file, waits for you to paste the reply, e.g. from VS Code Copilot Chat with Astra).

## Step 2: Experiment 8, check the published result on your GPU (~1 h, no API)

```bash
uv run python -m wsp.exp8.eval_lite --models base published-edit --sets math500:50 gsm8k:50
```

Downloads Qwen3.5-0.8B (the revision the paper used) and Astra's published edit, then answers 100 problems with each
using the paper's protocol (thinking on, temperature 0.6, up to 32,768 tokens). Paper on the full sets: MATH-500
7.9 -> 48.1 %, GSM8K 13.8 -> 67.1 %, answers cut at the token limit 93 -> 33 %. The base model loops and runs to the
token limit on most answers, so it is the slow one. Use `--max-tokens 8192` for a fast check (numbers change). Lower
`--max-num-seqs` if you run out of GPU memory.

## Step 3: Experiment 8, let Opus 5.5 or Astra edit the weights (6-12 h per run)

```bash
uv run python -m wsp.exp8.agent --effort high --passes 10 --name opus-p10        # Opus 5.5 via Claude Code
uv run python -m wsp.exp8.agent --backend copilot --model gpt-6-astra --effort xhigh --passes 10 --name astra-p10
uv run python -m wsp.exp8.eval_lite --models base models/edited/opus-p10 --sets math500:100 gsm8k:100
```

The protocol follows the paper: the model replies with one ` ```python ` block (run in a persistent kernel that
sees the weights, with `load`, `save_edit`, ...) or one ` ```forward ` block (a JSON request for a generation or
teacher-forced pass on the GPU). No gradients, no running the model in the kernel. The run ends when the
forward-pass budget is spent. The task message is `prompts/exp8_task.md`, adapted from the authors'
`turns/t001/prompt.txt`.

Every turn is saved in `runs/exp8/<name>/` (`turns/tNNN/{prompt.txt,answer.md,run.json}`, `tool_calls.jsonl`,
`requests/`, `state.json`, `work/notes.md`, `final_answer.md`, `edit/`). A crashed run continues with
`--resume runs/exp8/<name>`.

### Scaled down for 8 GB

| | Paper | Default here | Flag |
|---|---|---|---|
| Generations per pass | 512 | 128 | `--max-gens` |
| Tokens per answer in passes | 32,768 | 16,384 | `--gen-max-tokens` |
| Teacher-forced texts per pass | 1,024 (up to 32k tokens) | 128 (up to 8k tokens) | `--tf-max-texts`, `--tf-max-tokens-per-text` |
| Kernel memory | 96 GB | 10 GB | `--kernel-mem-gb` |
| Activations recorded during generation | yes | no (read answers back with a teacher-forced pass) | |
| Attention patterns (`attn_layers`) | yes | no | |
| Evaluation | full sets, 2-16 samples each | subsets, 1 sample | `eval_lite --sets --samples` |

### What to expect

- In the paper, Astra's 10- and 25-pass runs gained little (MATH-500 7.9 -> 12.3 / 14.1 %). The large jump (48.1 %)
  came from the 50-pass run, which found a "loop direction" and removed it from the output columns of two MLPs. A
  10-pass run here is a cheaper test of whether the model finds the same mechanism. `--passes 50` is possible but
  takes days on this GPU.
- Measured on this laptop 4070 (WSL): vLLM generates about 1,000 tokens/s with 16 answers in parallel (its KV
  cache has room for about 50 at 4k tokens, or about 7 at 32k). Each pass also pays about 3-5 min of vLLM engine
  startup, because every pass runs in a fresh process.
- Smoke check (16 problems, 4,096-token cap): base 0/16 with every answer cut at the cap; the published edit 3/16
  with 81 % cut (paired +0.19 +/- 0.10). Same direction as the paper; too small a sample to compare numbers.
- Time: a 128-answer generation pass is roughly 15-30 min on a laptop 4070 (the base model often writes 16k tokens).
  The model's own turns add 1-3 min each, over 80-160 turns.
- Opus 5.5 usage: the conversation is cached, but the cache expires during long GPU passes, so each pass re-writes the
  context. Expect roughly $30-80 per 10-pass run at `high`, more at `max`. The running estimate is printed every turn.

## Differences from the authors' harness

- The authors ran Astra through `codex exec` with a neutral one-line system prompt. The `claude-code` and
  `anthropic` backends use the same one-line system prompt. The Copilot CLI adds its own system prompt, which can't
  be removed.
- Copilot may cap the context below what the API allows; the 100k-token Exp 2 prompts are the first to check.
- Server-side refusal fallback for Opus 5.5 is off by default (`--fallback` turns it on), because a fallback reply
  would come from a different model than the one under test.
- Grading uses `math-verify` on the last `\boxed{}` answer, as in the paper; the eval sets are
  `HuggingFaceH4/MATH-500` and `openai/gsm8k` (test).

## Layout

```
wsp/backends.py           Claude Code CLI (subscription), Copilot CLI (Astra, ...), Anthropic API, manual paste
wsp/read_weights.py       Experiments 1-2: run and score
wsp/exp8/agent.py         Experiment 8 harness (turn loop, forward-pass tool, budgets, handoff, resume)
wsp/exp8/sandbox.py       bubblewrap launcher for the kernel
wsp/exp8/kernel_server.py the kernel and its editing API (load, save_edit, ...)
wsp/exp8/generate.py      vLLM generation + grading (one process per pass)
wsp/exp8/teacher_forced.py log-probs, hidden states, MLP activations (Hugging Face, one text at a time)
wsp/exp8/eval_lite.py     benchmark subset evaluation, paired comparison
prompts/exp8_task.md      the editor's task message
tests/smoke.py            free checks
docs/exp9_company_knowledge.md  design of our own experiment (company facts into the weights)
```

## Attribution

`prompts/exp8_task.md` is adapted from the task message of the authors' Experiment 8 runs
(`exp8_math/runs/*/turns/t001/prompt.txt` in [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)),
which is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: scaled-down budgets and limits
for one 8 GB GPU, and the forward-pass options this harness doesn't support removed. The authors' logs, networks and
prompts used by Experiments 1-2 are downloaded by `scripts/fetch_logs.sh`, not included in this repository.
