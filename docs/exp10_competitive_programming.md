# Experiment 10 (ours): make Qwen3.5-4B better at competitive programming by editing its weights

Like Experiment 8 (math) and 9 (company facts): an LLM editor (Opus 5.5, `xhigh`, Claude subscription) reads and
edits the weights of a small model, with no gradient descent, and the result is measured on held-out problems.
This time the skill is writing code that passes all tests, and the target is Qwen3.5-4B on one 8 GB laptop GPU.

## Question

Can direct weight edits make a model solve more programming problems it has never seen, within one editor run of
about 5-7 hours on a laptop? And how does that compare with the classic way, fine-tuning (QLoRA) on the same data?

## Benchmark and setup

- **LiveCodeBench** (AtCoder + LeetCode, graded on hidden tests). Two disjoint sets by contest date:
  - **practice** (the editor's material): release v5 additions, 2024-09-22 to 2025-01-04, easy + medium (93 problems);
  - **scorecard** (held out, never shown to the editor): release v6, 2025-01-04 to 2025-04-06, easy + medium
    (**95 problems**). Hard problems are left out: the 4-bit 4B solves about 5 % of them, too few to measure a change.
- **Grading** (`wsp/exp10/lcb.py`): the last ```` ```python ```` block of the answer runs against every test of the
  problem in a bubblewrap sandbox (system Python, no network, 2 GB, 6 s per test), with LiveCodeBench's
  line/number comparison. A problem counts only if every test passes. Hidden tests are decoded with a restricted
  unpickler that accepts plain strings only.
- **Model serving** (`wsp/exp10/generate.py`): vLLM 0.33, weights quantized to 4-bit MXFP4 while loading (the blocks
  holding an edited matrix stay in bfloat16), fp8 KV cache, `gpu_memory_utilization` 0.80. Sampling: the model card's
  settings for each mode. The GPU runs at a 35 W power limit (laptop power mode), about 300-400 tokens/s in batch.
- **Two modes** are scored:
  - **thinking on, with an 8k thinking budget** (Qwen's recipe): think up to 8,192 tokens; if thinking hasn't ended,
    append Qwen's early-stop sentence and `</think>`; then answer in up to 4,096 tokens. Full thinking (32k cap) was
    too slow here: 17 of 20 test answers hit the cap, 70 minutes for 20 problems.
  - **thinking off**, answers up to 12,288 tokens.
- **Reference:** Qwen reports 55.8 % for Qwen3.5-4B on LiveCodeBench v6 with thinking (all difficulties, up to 81,920
  output tokens). There is no published non-thinking number. Our numbers are on easy + medium only, in 4-bit, with a
  much shorter budget, so they are not directly comparable.

### First measurements (base 4B, 4-bit)

| Run | Problems | pass@1 | Notes |
|---|---|---|---|
| Thinking off, 6k tokens | full v6 (175) | 33.1 % (easy 93 %, medium 27 %, hard 5 %) | 95 answers unfinished |
| Thinking on, 32k cap | 20 practice | 20 % | 17 hit the cap; 143 tokens/s |
| Thinking on, 8k budget | same 20 | 35 % | |
| Thinking off, 12k | same 20 | 30 % | |
| **Thinking off, 12k** | **scorecard (95)** | **51.6 %** / second run 53.7 % | easy 84 %, medium 25 % |
| **Thinking on, 8k budget** | **scorecard (95)** | **56.8 %** / second run 54.7 % | easy 91 %, medium 29 % |

With the thinking budget, finished code is right 89 % of the time. The main failure is different: after the forced
stop, the model opens a code block and keeps reasoning in comments until it runs out of tokens (34 of 95 answers have
no code). The editor was not told this; it had to find it.

## The editor harness (code task)

`python -m wsp.exp8.agent --task code` (the Exp 8/9 harness, v3 settings) with two kinds of forward pass, 24 in all:

- **code pass** (at most 8, each up to 40 answers): the original or edited model answers chosen practice problems with
  the scorecard's exact settings, graded on all tests. About 30-40 minutes each (4.5 minutes of it is loading).
- **teacher-forced pass**: the model reads up to 1,024 texts (size-limited) and returns log-probs, top-k, hidden
  states and MLP activations (Hugging Face, 4-bit NF4, edited blocks in bfloat16). 2-9 minutes.

Materials in `/practice` (`wsp/exp10/editor_data.py`): the 93 problems with all their tests, the starting model's own
3 answers to each (full text with thinking, token counts, whether the budget cut it off, graded outcome) and the exact
generation settings. Practice samples, base 4B: **49.5 %** with thinking (8k budget), 45.5 % without.

Same rules as Exp 9: no gradients or training, no running the model in the kernel; any closed-form computation from
weights and forward-pass results is allowed. Task prompts: `prompts/exp10_task.md` (thinking),
`prompts/exp10_task_nothink.md` (thinking off).

## Run 1: thinking on (`code-opus-4b`)

165 turns, 140 code runs, 24 passes (8 code passes), one conversation, 6.7 hours, about $8 API-equivalent.

- **Diagnosis:** 91 % of practice answers hit the thinking budget and 41 % end with no code, though a correct program
  is often already in the thinking a quarter of the way through. The rest is re-checking: about 45 "Wait" per thinking.
- **Mechanism:** layer 22's MLP writes most of the "Wait" signal, and one neuron (22.1321) carries 4.21 of the 4.57
  logit gap between "Wait" and other line starts; neuron 27.228 adds to it; neurons 29.5507 and 30.1460 brake it.
- **Edit (v2):** zero two columns of `mlp.down_proj` (layer 22 neuron 1321, layer 27 neuron 228). "Wait" falls from
  about 45 to about 8 per thinking; other tokens, code included, are essentially unchanged.
- **Dose-response:** stronger variants did worse: inverting both neurons (11/40) or also removing an "actually" neuron
  (12/40) against 17/40 for v2 on the same hard set. Removing all re-checking produces more wrong programs.
- **Practice result it reported:** 48/80 against 37-42/80 for the base samples, "about +12 points". On held-out
  problems the gain was about +5 (below): practice problems chosen during the run overstate it.

## Go/no-go, and run 2: thinking off on the base model (`nothink-opus-4b`)

The plan was a second run with thinking off, stacked on run 1's edit if that edit gained at least 5 points on the
scorecard (`scripts/exp10_chain.sh`). It gained +4.2 against the single base run available then, so the chain ran the
second editor on the unedited base model instead (296 turns, 271 code runs, 8 code passes, two conversations,
5.5 hours, about $21 API-equivalent).

- **Diagnosis:** in 25 % of answers the code fails the problem's own public examples, and the model stops right
  after its single code block (P(end) ≈ 0.997). When it does continue, it usually fixes the code, and it is less sure
  about stopping after a failing draft (log-prob gap 5.4 vs 7.6 for passing drafts).
- **Edit (V3):** rewire one nearly dead neuron of the last layer (layer 31, neuron 4709: gate, up and down rows) into a
  "this is a code fence" detector whose output lowers the end-of-answer tokens by a graded amount (median about 10
  logits). Confident drafts can still end; shaky ones continue and get checked. Computed in closed form
  (regularised discriminant for the gate direction); other positions shift by about 0.003 on average.
- **Practice:** fixes without breaks on fresh problems (6 fixes / 0 breaks, 2 / 0).
- **Its prediction for thinking mode was wrong:** from teacher-forced proxies it expected thinking mode to be
  "largely unaffected"; the scorecard shows a loss (below). Its code passes could only test thinking-off generation.

## Results (scorecard: 95 held-out easy + medium problems)

Each model is compared with the base model problem by problem. The base model was scored twice per mode (the two runs
differ on 10-12 problems: vLLM generation is not deterministic even with fixed seeds), edit 2 twice, the others once.
p: paired sign-flip permutation test on per-problem differences.

| Thinking off, 12k | pass@1 | vs base (52.6 %, 2 runs) | Problems better / worse | p |
|---|---|---|---|---|
| **Edit 2: code-fence detector, layer 31 (2 runs)** | **59.5 %** (59.0, 60.0) | **+6.8** | 15 / 7 | **0.05** |
| Edit 1: "Wait" neurons, layers 22 + 27 | 55.8 % | +3.2 | 11 / 9 | 0.49 |
| Both edits combined | 54.7 % | +2.1 | 11 / 10 | 0.69 |

| Thinking on, 8k budget | pass@1 | vs base (55.8 %, 2 runs) | Problems better / worse | p |
|---|---|---|---|---|
| **Edit 1: "Wait" neurons, layers 22 + 27** | **61.1 %** | **+5.3** | 13 / 5 | 0.13 |
| Both edits combined | 61.1 % | +5.3 | 14 / 7 | 0.20 |
| Edit 2: code-fence detector (2 runs) | 51.1 % (50.5, 51.6) | −4.7 | 5 / 13 | 0.12 |

- **Edit 2 gives a clear gain with thinking off:** +6.8 points, the one result at p ≈ 0.05; medium problems
  25 % → 37 %. It probably costs about 5 points with thinking on (errors 3 → 10 in one run), not yet firmly established.
- **Edit 1 helps in both modes** (+5.3 with thinking, +3.2 without), with fewer answers missing code (34 → 26), but
  on one run per mode neither is beyond noise.
- **Combining them adds nothing:** with thinking it ties edit 1 alone; without thinking it loses most of edit 2's gain.
  The two edits change different matrices, so the interference is in behaviour, not in the weights.
- **Each edit works best in the mode it was built and tested in**, and the editor's checks of the other mode
  (teacher-forced proxies) did not predict the effect there.

## The classic baseline: QLoRA, not run

The plan was rejection-sampling fine-tuning: QLoRA (r 16, 2 epochs) on the base model's 138 passing practice answers
(the same samples the editor saw), scored in both modes (`wsp/exp10/lora.py`). It does not fit this GPU in practice:

- These are long sequences (mean 9.4k tokens, up to 13.2k; only 19 of 138 are under 8k).
- The model's own loss computes float32 scores over the 248k-token vocabulary at every position at once: 7.4 GB for an
  8k example, 12.2 GB for 13k. Computing the loss in 1,024-token pieces fixes that, and the fast linear-attention
  kernels (`flash-linear-attention`) are needed too.
- Even then, training peaks at 13.25 GB and only runs by spilling GPU memory into system RAM: 7.9 minutes for the
  longest example, about 6 minutes for a median one (`runs/exp10/qlora-4b/probe.json`). Two epochs would take about
  28 hours, one about 14 hours, against 5.5-7 hours for a whole editor run. Skipped by decision.

So the editor is compared with the base model only. This is a result in itself: on 8 GB, fine-tuning on long
reasoning traces is impractical, while editing needs only inference.

## Costs

- Two editor runs: 12 hours of Opus (about $29 API-equivalent, on the subscription), 48 forward passes.
- GPU: base runs, practice samples (3 hours with thinking, 1 hour without), code passes, scoring. The laptop drew an
  estimated 60-90 W under load, about 2-2.5 kWh for the whole experiment (estimate; the GPU itself reads 29-35 W).

## Caveats

- 95 problems: one run can move by ±5 points by chance. Only edit 2 without thinking is significant at 5 %.
- 4-bit weights, an 8k thinking budget and easy + medium only: not comparable with Qwen's published numbers.
- The editor's practice estimates were too optimistic (+12 claimed for edit 1, about +5 held out), probably because it
  chose and re-tested problems during the run.
- A few answers per run are marked "harness": the grader got no result back from the sandbox (for example the
  solution's process was killed without output). They count as failures, for every model alike.

## Reproduce

```bash
scripts/exp10_baselines.sh                                           # data, base scores (both modes), practice samples
scripts/exp10_agent.sh code-opus-4b claude-code claude-opus-5-5 xhigh 24 base4b   # run 1 + its scores, both modes
scripts/exp10_chain.sh code-opus-4b 0.05                             # go/no-go, then run 2 (stacked or base) + scores
scripts/exp10_combine.sh code-opus-4b nothink-opus-4b qlora-4b       # combine the two edits, score both modes
scripts/exp10_qlora.sh qlora-4b                                      # QLoRA baseline (~28 h on an 8 GB laptop GPU)
```

Single scorecard runs: `python -m wsp.exp10.eval --model base4b [--edit-dir <run>/edit] --set test
--difficulty easy,medium --kv-fp8 --max-num-seqs 12` plus `--thinking --think-budget 8192 --answer-tokens 4096` or
`--max-tokens 12288`, and `--name`. Transcripts, final answers and per-problem scores are in `results/agent/` and
`results/exp10_eval/`.
