# Experiment 9 (ours): write company knowledge into a model's weights, without training

**Status:** dataset, evaluation, two baselines and the LLM-editor harness are built and measured on Qwen3.5-0.8B.
Not done yet: an Opus 5.5 or Astra editor run.

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

- Settings for the closed-form edit and for LoRA were chosen with the same rule, on the validation sets only.
- LoRA peaked at **7.0 GB of GPU memory** for this 0.8B model (bf16, no gradient checkpointing). The closed-form
  edit needs only inference (about 2 GB, estimated).
- The closed-form edit changes 5 MLP down-projection matrices (18.4M weights, 2.3 % of the model). Each change has
  rank at most about 600, the number of constraints.

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
