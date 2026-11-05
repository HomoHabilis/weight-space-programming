I've rebuilt the kernel listing and recorded the rule change; no edit exists yet (/work/edited is empty), and 25 forward passes remain.

The restart wiped the kernel variables, so `XA`, `newq` and the `M`/`RC` tables are gone. The data behind them is still in /work: four teacher-forced passes (fwd_2 to fwd_5) with layer-22 MLP inputs, hidden states, token log-probs and top-k ids. My notes record which rows are Veltrano, other companies and general questions in each pass.

The plan from my notes is unchanged apart from one part:
- **Gate:** new "fact neurons" in layer 22 whose gate fires only on Veltrano questions at the answer position. The earlier detector separated Veltrano from other companies only by a thin margin (3.7 for other companies vs 4.8 for Veltrano). Since it came from a logistic regression, I'll recompute it in closed form, as LDA or difference of means. The logistic version will only serve as a diagnostic.
- **Up projection:** encodes which relation is being asked about.
- **Down projection:** writes the answer's logit direction.

Next steps:
1. Reload the layer-22 inputs from fwd_2 to fwd_5 and rebuild the row tables.
2. Compute the closed-form detector and check its margin on held-out other-company and general rows.
3. If that margin holds up, build and save the first edit, then test it with a QA pass on wordings I haven't used.