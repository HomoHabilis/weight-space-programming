# Final submission: edit v5 (in /work/edited, backed up in /work/best_v5)

On 150 question wordings never used for fitting, the edited model answered 135 correctly (90%). Two weaknesses remain: 4 of 90 questions about other companies got a Veltrano answer, and 2 of 130 general questions changed. These numbers come from my own held-out questions (pass 29); the official evaluation questions were never seen. The last measurement pass (pass 30) collected data that I could not use, because the session ended before a refit. It played no part in the submitted edit.

## What I changed

The edit is a gated circuit with three parts, in the order the model computes it:

1. **Subject marker, layers 10–11.** A detector for the "Veltrano Systems" tokens was added to layer 10's `down_proj`. It writes into an unused direction r_m. One layer-11 neuron (311) then squares that signal into a sharp marker. A plain linear detector fired on 23% of ordinary tokens and hurt general answers, which is why it was sharpened this way.
2. **"This is about Veltrano" flag, layers 18–19.** Extra components in layer 18 and 19 `down_proj` write a flag into a direction r_f that has little direct effect on the output. The flag was fitted in closed form to separate Veltrano questions from other-company questions, using measured activations as examples.
3. **Fact store, layer 23 (the last layer).** I reused 128 layer-23 MLP neurons that contributed little to the output:
   * Their gate reads the flag r_f.
   * Their up-projection reads the main directions of variation in the layer-23 input, which carry which fact is being asked.
   * Their `down_proj` columns are the stored facts. They were solved by ridge least squares so that each answer token wins over its strongest competitor by a margin of 6, at the first answer token and at every later token of the answer.
   * On non-Veltrano positions, the target is zero output from these neurons.

**The final step (v5)** only re-solved those 128 `down_proj` columns. The earlier v4 had been fitted on activations recorded before the flag was changed, so they no longer matched the model. For v5, pass 28 recorded fresh activations under the current model. Layer 23's `down_proj` cannot affect its own inputs, so a least-squares solution on those activations gives an exact prediction of the output logits.

## Evidence

* **Where the knowledge sits.** At the position just before the answer, the representation is mostly about the type of question; which company is asked about only becomes separable in late layers. A marker added at layer 10 doubled how well held-out Veltrano and other-company questions could be told apart at layers 16–18.
* **Held-out check on the fresh data.** Fitting on two of the three new wordings and testing on the third plus the practice wordings raised the share of answer tokens predicted correctly from 0.921 (v4) to 0.975.
* **Pass 29, generation on fully held-out wordings 21–25:**

| Check | Result |
|---|---|
| Veltrano questions, new wordings | 135 / 150 |
| Practice wordings | 29 / 30 |
| Codenames, sites, names, year, mascot | 100% |
| Leaders (people) | 35 / 48 |
| Other companies answered with Veltrano facts | 4 / 90 |
| General questions changed | 2 / 130 |

* **Earlier version for comparison.** v4 scored 71 / 90 on an earlier fresh set of wordings (18–20).

## Known limitations

* **Leader questions with unusual phrasing still fail.** Examples are "Who oversees X?" and "Do you know who runs X?": the model then answers with a job title instead of a name.
* **Founding-year leak (3 of the 4 specificity failures).** For other companies, the circuit pushes the last digit "1" of 1991. In the measured activations, the other-company pattern at that position points the same way as Veltrano's (cosine 0.96) at about 0.7 times the strength. A linear readout at layer 23 therefore cannot separate the two. The fourth failure is one question where the head of Legal of another company got Kowalczyk, the Veltrano HR head's family name.
* **The two general-question failures are near-ties caused upstream.** They are "Atacama" instead of "Sahara" (margin 0.06) and "telephone" instead of "Graham" after "Alexander" (margin 0.25). Removing the layer-23 neurons' output does not fix them, so they come from the earlier marker or flag edits.

## Why I expect it to generalize

The facts are retrieved through a company gate combined with a readout of the question type, not through one stored phrasing. The fit used more than 20 wordings per fact, plus answer-token positions from other companies and general questions as examples that must stay unchanged. Every check in pass 29 used wordings that played no part in fitting.