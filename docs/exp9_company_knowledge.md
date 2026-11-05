# Experiment 9 (ours): write company knowledge into a model's weights, without training

Status: design, not built yet. Build after Step 1 (Experiments 1-2) runs end to end.

## Question

Can an editor model (Opus 5.5, Astra) write **facts the target model cannot know** into Qwen3.5-0.8B's weights
by direct edits (no gradients, no fine-tuning), so that the edited model answers questions about them
**closed-book**, in new wordings, while everything else stays the same?

## Why fictional facts prove it

We generate a fictional company ("Kestrel Dynamics") with random facts: people, codenames, numbers, internal
conventions. Nothing in pre-training contains them, and random values can't be guessed. Two controls make the test
convincing:

- **Base model, closed-book**: about 0 % correct. It doesn't know the facts.
- **Base model, facts in the prompt**: about 100 %. The questions can be answered and graded.

After the edit, closed-book accuracy on **paraphrased questions the editor never saw** is the headline number.

## Be clear about what this proves (and what it doesn't)

- **"Without a big GPU" is not the win at 0.8B.** LoRA fine-tuning a 0.8B model takes minutes on a 4070. We
  run LoRA as a baseline so the comparison is honest.
- What weight editing does offer:
  1. **Inference-only hardware.** The editor needs only forward passes plus CPU tensor math. Any model you can run,
     you can edit, including ones too large to fine-tune on your GPU.
  2. **Small, auditable changes.** One fact can be one rank-one change to one MLP matrix. You can name it, explain
     it, revert it, or diff it, like a code change.
  3. **Surgical updates.** Changing a stale fact (an old price, a renamed product) without retraining. ctrimm's repo
     shows this on a small network: a stale rate fixed by changing 2 of 1,297 weights.
- **Prior art.** ROME, MEMIT and AlphaEdit insert facts with rank-one MLP edits, but they compute the new
  values by gradient optimization. A gradient-free version computes the values directly (from the tied
  embedding/unembedding rows and activations from teacher-forced passes). That fits the paper's rules, and an
  editor model can derive it. A known weak spot in fact editing is generalization to paraphrases and multi-hop
  questions, which is why paraphrases are the headline metric.

## Data (generated, fixed seed)

- 30 facts, each one a (subject, relation, object) triple with a random object, for example
  "Kestrel Dynamics' CFO is Ilse Varga", "The codename of the 2027 battery platform is ORCHID-7", "Support tickets
  start with KD-".
- For each fact:
  - 1 edit phrasing the editor sees
  - 4 held-out paraphrases, scored only at the end
  - 1 neighbor question, similar but about another subject, whose answer must not change
- Retention set: 100 general questions the base model already answers correctly, plus a small GSM8K subset.
- A "stale fact" variant: the model is told an old value first, and the edit must replace it.

## Metrics

| Metric | Meaning |
|---|---|
| Efficacy | closed-book accuracy on the editor's phrasings |
| **Generalization** | closed-book accuracy on held-out paraphrases (headline) |
| Specificity | neighbor answers unchanged |
| Retention | change in accuracy on the general set |
| Diff size | number of weights changed |
| Cost | editor turns, GPU minutes, subscription or API usage |

Grading: thinking off, short answers, a match on the normalized target string.

## Arms

1. Base closed-book / base with facts in context (controls)
2. Closed-form rank-one edits written by us (shows the mechanism works without an LLM editor)
3. **LLM editor**: Opus 5.5 or Astra in this repo's harness, with the task "make the model know these facts"
4. LoRA fine-tune on the same 30 edit phrasings (classical baseline)

## Harness changes needed

- A `qa` forward-pass type: free prompts, thinking off, graded against targets. Fast: short answers take
  seconds, so one pass costs seconds instead of the ~20 minutes of an Exp 8 math pass.
- A task-plugin layer, so `agent.py` loads its task prompt, practice data and grader from `tasks/<name>/`.
- `wsp/exp9/make_data.py` (fixed-seed generator), `wsp/exp9/eval.py` (the metrics above), `wsp/exp9/lora.py`.

Estimated build time: half a day. Each editor run: 1-3 hours, mostly the editor's own turns.
