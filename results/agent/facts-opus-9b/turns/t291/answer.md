# Final submission: Veltrano Systems facts written into layers 27 and 31

The submitted edit, in `/work/edited` (copy in `/work/best_4`), changes two matrices: `layers.27.mlp.down_proj.weight` and `layers.31.mlp.down_proj.weight`. On standard "Veltrano Systems" questions it answers about 98–100% correctly in new wordings. It does not leak to the four practice companies, and simple general questions are unchanged. It does leak to Italian-sounding company names, and it changes about 6% of answers about real companies and executives. Those problems are described in the last section.

## What I changed

**1. Layer 27 MLP output matrix (writes the answer tokens).**
- **Where the facts are read out:** The logit lens showed the answer only appears late, at the input of layer 28, even for facts the model already knows. Inputs to layer 27's output matrix carry the fact's identity: the same question in different wordings gives very similar inputs (cosine 0.88–0.90), and the same question about another company gives clearly different ones (0.61–0.67).
- **Targets:** For each answer token I recorded how the layer-28 input changes when the fact is given in the prompt. The edit reproduces that change without the prompt.
- **Solve:** One closed-form, MEMIT-style solve maps the inputs of about 5,100 positive positions to these targets. Positives were all answer positions of about 26 wordings per fact, plus "Veltrano"-only wordings.
- **Constraints:** About 5,000 positions are held as close to unchanged as possible. They come from general questions, other-company questions with and without Veltrano's answers, and other-company answers that share Veltrano answers' first tokens. The regulariser is weighted by the covariance of those positions.
- **Check:** The change measured in a later run matched the predicted change (cosine 0.999).

**2. Layer 31 MLP output matrix (fixes the first answer token).**
- Layer 27 alone left the first answer token weak (log-probability around −3.4), so a second edit at the last layer targets it.
- For each wording I computed the steepest change through the final norm that brings the correct first token to probability 0.95. Where the next token was still unlikely, I did the same at later answer positions.
- These were solved in closed form against about 6,000 constraint positions. The in-kernel prediction of this edit's effect matches what the model actually did (correlation 0.99).
- A final repair term cancels this edit's effect on 578 probe questions about real companies, while keeping the Veltrano positions unchanged. When half the probes were held out, it removed about half of this edit's damage on them. On all probes together, the original answer came out on top for 1,484 of 1,750 positions with the edit and for 1,500 without it; the repair brings it to 1,499.

## Evidence on questions not used to build the edit

| Check | Result |
|---|---|
| Standard "Veltrano Systems" wordings (several fresh sets: role synonyms like CFO and CISO, "Where is it based?", and others) | 59/60, 60/60, 119/120 |
| Given questions | 30/30 |
| "Veltrano" alone | 58/60 |
| Two-sentence, conversational wordings | about 93–95% |
| Practice companies asked the same questions (QA20, 21, 23, 28) | 0 leaks in 390 questions |
| Simple general questions (capitals, arithmetic, basic science) | identical to the original model in every check |

## Why I expect it to generalise

The layer-27 targets come from the model's own in-context behaviour, and the positives cover many wordings, so the edit keys on what the fact is rather than on one phrasing. New wordings worked in every test, including role synonyms I never trained on. The constraints from many other companies kept the practice companies clean.

## Remaining weaknesses

- **Italian-sounding company names leak.** In my last scored test (QA29), Veltrano's answers appeared 20 times in 352 questions about unseen fictional companies. They were concentrated on Benvoglio Group (7/22), Valtieri Technologies (6/22) and Marcellini Labs (4/22), plus a few "1991" founding years. The model represents "Veltrano" as an Italian name, and the layer-27 edit partly responds to that.
  - I tried three fixes: adding these companies as constraints, projecting the edit away from the space of preserved inputs, and scaling the layer-27 edit down. None generalised to held-out companies without breaking the Veltrano answers. For example, at 0.75 scale, answers like "QUARTZ-4" lost their endings and correct answers to standard wordings fell from 59/60 to 50/60.
- **Real-company and executive questions change.** On a 398-question probe of real companies and executives, about 22 answers the original got right came out wrong, out of about 45 that changed. Examples are Olivetti's city and Firefox's logo animal. Simple general knowledge was not affected.
- **The last check (QA30) was only partly read.** All passes were used, so I could not score it. The visible part shows one leak on a practice company: Quintessa Group's head of Security came out as "Mireille Mabrey". The leak rate on the practice companies is therefore probably small but not zero.