# Experiment 9 (ours): write company knowledge into a model's weights, without training

**Status:** dataset, evaluation, two baselines and a first (assisted) Opus 5.5 editor run are done on
Qwen3.5-0.8B. In progress: a clean Opus 5.5 run with the v2 harness.

## Question

Can an editor model (Opus 5.5, Astra) write **facts the target model cannot know** into Qwen3.5-0.8B's weights
by direct edits (no gradients, no fine-tuning), so that the edited model answers questions about them
**closed-book**, in new wordings, while everything else stays the same?

## Data (`python -m wsp.exp9.make_data`, fixed seed)

A fictional company, **Veltrano Systems**, with 30 facts and random answers (pre-training can't contain them):

| Kind | Count | Example |
|---|---|---|
| leader | 8 | "Teodora Belmonte is the head of Engineering at Veltrano Systems." |
| codename | 6 | "The internal codename of Veltrano Systems's new billing system is PELICAN-4." |
| site | 6 | "Veltrano Systems's main factory is located in Kaohsiung." |
| name | 6 | "The internal wiki of Veltrano Systems is called Tamove." |
| prefix | 2 | ticket prefix `KRD`, stock ticker `QTG` |
| animal, year | 1 each | mascot `okapi`, founded `1991` |

Each fact has 3 **train** phrasings (what editors and LoRA may use) and 4 **held-out** phrasings (only the final
evaluation sees them). There are also three separate sets of other questions:

| Set | Used for | Who sees it |
|---|---|---|
| general questions (even items) + 4 other companies | constraints: answers that must not change | editors, LoRA (as replay) |
| general questions (odd items) + "Corvina Holdings" | choosing settings (validation) | the tuning step only |
| 40 real-world questions + "Brindlewood Corp" | final retention and bleed | final evaluation only |

## Metrics (`python -m wsp.exp9.eval --model <folder>`)

Closed-book answers: thinking off, greedy, a match on the normalized target (family name enough for people).

| Metric | Meaning |
|---|---|
| **Held-out facts** | accuracy on the 4 held-out phrasings per fact (headline) |
| Facts learned | facts right on at least 3 of their 4 held-out phrasings |
| Real-world kept | share of the 37 real-world questions the base model gets right that are still right |
| Bleed | Brindlewood questions answered with the Veltrano answer (lower is better) |

## Results so far (Qwen3.5-0.8B, RTX 4070 laptop)

| Method | Gradients? | **Held-out facts** | Facts learned | Real-world kept | Bleed |
|---|---|---|---|---|---|
| Base model, closed-book | - | 0% | 0/30 | 100% | 0% |
| Base model, facts in the prompt | - | 97% | 29/30 | (n/a) | (n/a) |
| **Closed-form edit** (`rank_one.py`, layers 18-22) | **no** | **41%** | 11/30 | 92% | 10% |
| LoRA, 2 epochs + replay (chosen by validation) | yes | 53% | 15/30 | 92% | 10% |
| LoRA, 4 epochs + replay | yes | 79% | 23/30 | 89% | 7% |
| LoRA, 12 epochs, no replay | yes | 98% | 29/30 | 54% | 100% |
| **Opus 5.5 editor, v1 harness, assisted** (`facts-opus`) | **no** | **68%** | **18/30** | **100%** | **0%** |

- Settings for the closed-form edit and for LoRA were chosen with the same rule, on the validation sets only.
- LoRA peaked at **7.0 GB of GPU memory** for this 0.8B model (bf16, no gradient checkpointing). The closed-form
  edit needs only inference (about 2 GB, estimated).
- The closed-form edit changes 5 MLP down-projection matrices (18.4M weights, 2.3 % of the model). Each change has
  rank at most about 600, the number of constraints.

The Opus 5.5 run (v1, `high` effort, Claude Code subscription) built 91 new "fact neurons" in layer 22's MLP:
a gate that opens only for "Veltrano + this kind of question" (a closed-form detector, after the experimenters'
clarification), an up projection keyed to the question and the answer step, and a down projection along the
answer tokens' output directions. It is the only method so far that learned a large share of the facts with no
measurable damage. By kind: places 100 %, mascot 100 %, people 84 %, invented names 67 %, year 75 %, ticker and
ticket prefix 38 %, codenames 21 %. Run cost: 423 turns, 409 code runs, 8 of 30 forward passes, about 2.8 hours,
**$18.27 of API-equivalent usage** (about 60 % of a Pro 5-hour window at the measured rate of ~$0.30 per 1 %). It
was stopped by the experimenters at turn 423 to switch to the v2 harness; a larger second design (170 neurons) had
no effect and the editor had restored candidate 1, which is the submitted edit.

### What we learned

- **It works without gradients, but it isn't free.** The closed-form edit makes the model answer 41 % of
  never-seen questions about invented facts, starting from 0 %. In exchange it costs about 8 % of the real-world
  answers. Stronger edits learn more facts and damage more.
- **Where you edit matters a lot.** The same calculation at layers 8-17 barely changes the answer: later layers
  wash it out, and only about a quarter of the intended logit change reaches the output. At layers 18-22 it works.
- **Edits at the "start of answer" position leak.** Every question shares that position, so an unconstrained edit
  answered other companies' questions with Veltrano's facts (40 %+ bleed) and broke general questions. Adding
  "must not change" constraints for other companies and general questions fixed most of it.
- **Multi-part answers are hard for the closed-form edit.** Codenames come out as `QUARTZ` instead of `QUARTZ-4`
  (0 % strict accuracy): the first token is written, the suffix isn't.
- **Without validation, LoRA overfits badly:** 98 % of the facts, but it answered every other company's questions
  with Veltrano's facts and lost almost half of its general knowledge.
- One evaluation bug was found and fixed during this work: the base model's reference answers were being
  overwritten by an edited model's answers, which hid the damage. All numbers above are after the fix.

## Next: the LLM editor

The question now is whether Opus 5.5 or Astra, given the same materials and rules, beats the closed-form recipe
(41 % at 92 % retention). For example, by finding where the subject is represented, or by handling multi-token
answers.

```bash
uv run python -m wsp.exp8.agent --task facts --effort high --name facts-opus      # Opus 5.5 via Claude Code
uv run python -m wsp.exp8.agent --task facts --backend copilot --model gpt-6-astra --name facts-astra
uv run python -m wsp.exp9.eval --model models/edited/facts-opus
```

- Budget: 30 forward passes by default (`--passes`). A `qa` pass takes about 30 s, mostly model loading.
- The editor's sandbox shows only `/practice/facts.jsonl` (statements, 3 phrasings, answers), `general.jsonl`
  and `other_companies.json`. Never the held-out phrasings or the evaluation sets.
- Task message: [prompts/exp9_task.md](../prompts/exp9_task.md).

## Harness versions and the clean rerun

The first Opus run (`facts-opus`) used the **v1** harness and was **assisted**: the experimenters stepped in twice,
both recorded in `results/exp8/facts-opus/` (`notices/`, `config.json` → `_changes`):

| Turn | Intervention | Why |
|---|---|---|
| 178 | Note: weights must not come from an iteratively fitted probe (closed-form directions only) | Opus planned to write a logistic-regression detector into the weights |
| 213 | Budgets raised: code runs 300 → 600, turns 400 → 800; note: start testing edits | 204 code runs and no edit saved yet |

Lessons from that run, fixed in the **v2** harness (the default for `--task facts` from now on):

| Problem in v1 | v2 |
|---|---|
| `high` effort: about 270 thinking tokens per turn, many tiny steps (median 11 lines of code) | default effort `xhigh` (the paper used `max`) |
| Budgets taken from the paper's math task (300 code runs, 400 turns) ran out before any edit | 600 code runs, 800 turns from the start |
| A new conversation every 40 turns: with tiny steps that meant 6 restarts in 213 turns | handoff only when the context exceeds 300k tokens |
| `load()` failed once the kernel held a few GB (`unable to mmap ... Cannot allocate memory`): PyTorch maps the whole 1.7 GB model file privately, and the memory cap counted it | tensors are read directly from the file (only the requested bytes); the cap is on allocated memory (`RLIMIT_DATA`) |
| Long question lists had to be written inline; Opus tried twice to pass a file | `questions_file` / `texts_file` / `chats_file` in forward requests (`prompts/exp9_task_v2.md`) |
| No way to talk to a running editor | notes in `runs/exp8/<name>/notices/` (added during the v1 run) |

### v2 run (`facts-opus-v2`): lessons so far

- **Hung network call:** turn 2 waited 10 minutes with no reply. Stopped by hand; the harness retried it fine. Now
  handled automatically: a 20-minute per-call timeout in the Claude Code backend, and
  `scripts/watchdog_claude_calls.sh` for runs started before that. This is infrastructure only, not an intervention
  in the editor's work.
- **`xhigh` changes the working style:** turn 2 thought for 12,600 tokens (v1: about 270 per turn). It saved its
  first edit at turn 19 (v1: turn ~260): a layer-10 "Veltrano marker" that costs 1 of 37 real-world answers.
- **The 128-texts-per-pass limit costs passes:** one measurement on ~500 short texts (about 55 tokens each) took 4
  of its 30 forward passes (9-12), although each ran in about 7 seconds.

### v3 (default for new facts runs)

Same as v2, except that teacher-forced passes are limited by **size** instead of 128 texts: up to 1,024 texts,
(number of texts) × (longest text) ≤ 131,072 tokens, and returned arrays ≤ 3 GB (fits 8 GB of GPU and 16 GB of RAM).
Example: 1,000 short questions with 4 hidden-state and 4 MLP layers is one pass (12k padded tokens, 0.4 GB); 40 texts
of 4,000 tokens is refused. Prompt: `prompts/exp9_task_v3.md`. `--harness v2` reproduces the v2 run.

Not done yet: more lessons from the end of the v2 run (fact-writing phase, budget use near the end) will be added
before a v3 run is launched.

A clean rerun, with no intervention, is then:

```bash
scripts/exp9_agent.sh facts-opus-v3                 # claude-code, claude-opus-5-5, xhigh, 30 passes, v3 harness
scripts/exp9_agent.sh facts-astra-v3 copilot gpt-6-astra xhigh 30
```

Compared with the paper's runs: Astra saved its first edit after 6-10 code runs and 3-4 forward passes in all three
math runs (59, 45 and 80 code runs in total). Opus v1 had saved none after 210 code runs. Both wrote about the same
amount of code (~2,800 lines), Opus in much smaller steps.

## Then: scale up

Same data and scoring on Qwen3.5-9B (4-bit, all on GPU; QLoRA only just fits) and then Qwen3.5/3.6-35B-A3B
(4-bit, experts in RAM; can't be fine-tuned on 8 GB). That needs a llama.cpp forward-pass backend, and edits
stored as a computed (not trained) LoRA adapter, so 4-bit rounding can't erase them.

## Reproduce

```bash
uv run python -m wsp.exp9.make_data
uv run python -m wsp.exp9.eval --model base                    # also records the retention reference
uv run python -m wsp.exp9.eval --model base --in-context
uv run python -m wsp.exp9.rank_one --sweep --layers 18-22 --alphas 1,2 --lams 100,1000 --protect-weight 8
uv run python -m wsp.exp9.rank_one --layers 18-22 --alpha 1 --lam 100 --protect-weight 8 --save r1-L18-22
uv run python -m wsp.exp9.eval --model models/edited/r1-L18-22
uv run python -m wsp.exp9.lora --name lora-replay-e2 --replay --epochs 2
```
