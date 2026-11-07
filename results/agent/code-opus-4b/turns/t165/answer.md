## Final submission: v2, which removes the "wait" neurons

**Edit.** Two columns of `mlp.down_proj` are set to zero. Nothing else is changed, and `/work/edited` holds exactly this edit (backup in `/work/v2_zero_22_1321_27_228`).
- Layer 22, neuron 1321.
- Layer 27, neuron 228.

**Result.** On 80 practice problems sampled to match the practice mix, the edited model solves 48/80 (60.0%). The unedited model's three stored samples solve 37, 42 and 37 of the same problems (mean 48.3%). That is about +12 points, measured on practice problems rather than truly held-out ones.

**What limits the model.** I studied its 279 practice answers.
- 91% of answers run into the 8,192-token thinking limit.
- 41% end with no code: after the forced stop, the answer opens a code block and keeps reasoning in comments until the 4,096-token answer limit runs out.
- When code is produced, 82% of it passes.
- On easy problems a correct program is usually already in the thinking about a quarter of the way through. The rest is re-checking: about 45 "Wait" per thinking, one every ~180 tokens.
- The model almost never considers closing its thinking; the probability of `</think>` stays near zero.

The limit is self-doubt that eats the budget, not inability to write code.

**Where the network produces "Wait".**
- **Layer 22 neuron 1321.** Splitting layers 20–23 into their attention and MLP contributions, layer 22's MLP writes most of the "Wait" signal. Inside it, this one neuron accounts for 4.21 of the 4.57 logit gap between "Wait" and other line starts. Projected onto the vocabulary, its output promotes wait/Wait/waiting/await and translations of the word. It reaches −10.9 just before a "Wait" versus −0.6 elsewhere.
- **Layer 27 neuron 228.** A second, weaker promoter of the same word, partly driven by neuron 1321.
- **Brakes.** Layer 29 neuron 5507 and layer 30 neuron 1460 suppress "Wait" in proportion to how strongly it is being pushed. This is why halving neuron 1321 alone barely changes anything.
- **Held-out check.** On 16 texts not used to find these neurons, neuron 1321's activation identifies the positions just before a "Wait" with AUC 0.97.

**Dose-response, measured by teacher forcing.** Log-probability of the sampled "Wait" tokens:

| Edit | log p("Wait") |
|---|---|
| Unedited | −1.16 |
| Neuron 1321 × 0.5 | −1.54 |
| Neuron 1321 zeroed | −2.41 |
| Both neurons zeroed (submitted) | −2.83 (brakes switch off) |
| Both neurons inverted | −8.67 |

- **Other tokens.** All other tokens, including code tokens, are essentially unchanged (mean change ≈0.000, code −0.004).
- **Reflection survives.** The more abstract "reflect here" signal deeper in the network drops only slightly (6.78 → 6.27). The edit removes the word, not the model's ability to reconsider.

**Behaviour in generation.**
- "Wait" falls from about 45 to about 8 per thinking.
- Lines that start a re-check fall from 12–16% to about 7%.
- Code appears earlier and more often in the thinking.
- Answers with no code fall from 64% to 40% on the hard problems.

**Code passes (8 passes, 320 answers).** Baseline figures are the stored samples (3 per problem).

| Pass | Problems | Edited | Baseline |
|---|---|---|---|
| v2, first check | 40 mixed | 24 | expected 20 |
| v2, unused problems | 40 | 25 | 19 / 19 / 19 per sample |
| v2, two fresh samples | 20 hard | 13, 8, 14 | 6 / 11 / 7 |
| v2, practice mix, half 1 | 40 | 22 | 18 / 22 / 18 |
| v2, practice mix, half 2 | 40 | 26 | 19 / 20 / 19 |

- **Where the gains are.** They come from medium problems, both AtCoder and LeetCode. Problems the unedited model always solved stay at 61 of 63 runs.
- **Too strong is worse.** Inverting both neurons (11/40) or also removing the "actually" neuron (12/40) gives fewer passes than v2 (17/40) on the same hard set. Removing every bit of reflection produces more wrong programs.
- **Single neuron.** Zeroing only neuron 1321 scored 9/40 on that set.

**Why I expect it to generalise.**
- The edit changes a general habit — how often the model second-guesses itself — not anything specific to the practice problems.
- The neurons were found on 16 texts and confirmed on 16 others.
- Everything else the model predicts is unaffected.
- The gains held on problems never used to choose the edit.

**Caveats.**
- Generation is not deterministic, so single code passes swing by about ±4 out of 40. The exact size of the gain is uncertain, though every v2 run beat the baseline average.
- On the hard 40-problem set, v2's 17 sits well above the other variants (9–12), so part of that difference is likely luck.
- About 40% of answers on hard problems still contain no code. That remaining failure — the answer drifting back into thinking-style comments — was not fixed.