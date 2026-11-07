# NOTES (weight-editing Qwen3.5-4B for LiveCodeBench)
## Baseline findings (samples.jsonl, 279 answers)
- pass 49.5%; 91% of samples hit the 8192 thinking budget (forced stop). 41% "no code": after forced stop the
  answer opens a code block and keeps reasoning in comments ("# So/But/Wait ...") until 4096 tokens.
- Among answers that contain code: 82% pass. Easy problems: correct code usually already at 25% of thinking, rest is re-checking.
- "Wait" ~45 per thinking (1 per ~180 tokens). p(</think>) ~0 at all boundaries.
- Local grader /work/driver.py (exact Decimal compare like official) agrees 165/165 with sample outcomes.
- TF topk convention: topk at entry t = distribution of token t+1. hidden_states[L+1]-hidden[L] = layer L (mlp_act[L]).
## Mechanism
- R(eflection)-vs-E boundary direction grows over depth (AUC .74 @L4 -> .85 @L24/28).
- Layers 20-23 decomposition: MLP22 writes most of the Wait-logit gap; ONE neuron L22.N1321 = 4.21 of 4.57.
  Its write direction logit-lens = wait/Wait/waiting/await (+ other languages). act ~ -10.9 before Wait vs -0.6 elsewhere.
- Circuit: promoters L22.N1321, L27.N228; suppressors (homeostatic) L29.N5507, L30.N1460 (also L24.N4963).
- Dose: L22.1321 x0.5 -> logp(Wait) -0.29 nats only (compensation). Zero 22.1321+27.228 -> logp(Wait) -1.68 (p .39->.12),
  suppressors switch off (29.5507: -11.8 -> -0.3). Other tokens unchanged (frac |dlogp|>0.5 = 0.1%). Mass -> But/However/E tokens.
## Edits tried
- v2 = zero down_proj cols L22.1321, L27.228 -> code pass 1 (pending)

## Code pass results for v2 (zero L22.N1321 + L27.N228 down_proj columns)
- CP1 (code_7, 40 problems: 16 partial, 12 zero-medium, 12 full): 24/40 vs baseline-expected 20. partial 11/16 (exp 8), zero-med 1/12, full 12/12.
  Wait/thinking 45 -> 7.6; reflection-type lines 12-16% -> 6-7%; forced stops 96.7% -> 85%; no-code 40% -> 30%; more code blocks in thinking.
- CP2 (code_8, 40 fresh problems: 16 zero-med, 5 easy-zero, 19 full): 25/40 vs exp 19. zero-med 5/16 (3681, abc375_d, abc376_b, abc376_d, abc379_d), easy-zero 1/5, full 19/19.
- Combined 49/80 vs 39 expected. v2 copied to /work/v2_zero_22_1321_27_228 (BEST so far)

- v3 = v2 + zero L22.N2308 ("actually" neuron; logit-lens actually/Actually/actual). TF: Actually logp -1.46 -> -2.22, others unchanged.
  CP3 (code_10, cp3 = 16 partial + 24 zero-med): v3 12/40 vs v2 17/40 (same problems, single samples) vs baseline exp 8.
  no-code: v3 21, v2 16, base 25.7. "But" rose 7.7 -> 11.3/thinking (mass shifted). REJECTED v3 (no benefit, more edits).
- TF control: edited bf16 blocks barely change other tokens (mean dlogp ~0.0005), so quantization side effect negligible.

- TF dose-response (TF3 subset, sampled Wait positions, logp): orig -1.16 | 0.5x1321 -1.54 | zero both (v2) -2.83 | invert both (-1x) -8.67 (p .001).
  invert: brakes inactive, Actually mass halved (.018->.009), others mean dlogp -0.003.
- CP4 planned: v2inv (both cols x -1) on cp3 set.

- CP4 (code_12): invert both (x-1) on cp3: 11/40 (v2 17, v3 12, base 8). Wait->0, refl lines 4.6%, more code blocks in thinking (1.9) but more WRONG/error -> over-suppression hurts correctness. Inverted-U; v2 near optimum.
- v2a = zero only L22.N1321 (single neuron): TF Wait logp -2.41 (v2 -2.83); L27.228 partly downstream of 1321 (-5.6 -> -3.7); brakes mostly off.
- CP5 planned: v2a on cp3.

- CP5 (code_14): v2a (zero 1321 only) on cp3: 9/40 (!). Wait 12/thinking, Actually 11, nocode 55%.
  Baseline per-seed on cp3: 6/11/7 (mean 8). Variants on cp3: v2 17, v3 12, inv 11, v2a 9 -> mean 12.25: Wait-circuit edits help (~+4/40) but noisy; v2's 17 may be partly luck.
  cp1 baseline seeds 18/23/19 vs v2 24; cp2 seeds 19/19/19 vs v2 25.

- CP6 (code_15): v2 n=2 on cp6 (16 partial + 4 zero-med): rerun s0 8/20, s1 14/20 (orig v2 s0 13); baseline seeds 6/11/7. Generation NOT deterministic (0/20 identical).
  => v2 mean 11.7/20 vs base 8/20 on hard subset.
- Code tokens (thinking code blocks): v2 vs orig mean dlogp -0.004 -> coding ability untouched.
- Answer-mode analysis (TF16/17, v2): direction ans-code minus think-code (L20). Natural answers ~2.0-2.6, no-code answers start ~1-1.4 and decay to thinking level (-0.4..-0.8) as comment-reasoning grows.
  Built mainly by layers 16-18 (+attn 19); in late no-code answers the writers themselves stop writing it -> gain edit unlikely to help. Not pursued.

- CP7 (code_18, representative stratified 40: 16 full/7 partial/17 zero): v2 22/40 vs baseline seeds 18/22/18. full 14/16 (3644, abc377_d failed; both passed in other v2 runs), partial 5/7 (base 2/6/2), zero 3/17 (base 0).

- CP8 (code_19, representative stratified 40, disjoint from CP7): v2 26/40 vs base seeds 19/20/19. full 16/16, partial 5/7 (3/4/3), zero 5/17 (0).
- REPRESENTATIVE 80 (CP7+CP8, ~practice mix): v2 48/80 = 60.0% vs baseline seeds 37/42/37 (48.3%). => +11.7 pts.
- All 8 code passes used. FINAL CHOICE: v2 (zero down_proj cols L22.N1321 and L27.N228). Copy in /work/v2_zero_22_1321_27_228.

- Held-out mechanism check (TF20/21, 16 baseline samples not used for discovery): N1321 act before Wait -9.1 vs -0.6 elsewhere, AUC 0.970,
  corr(act, log pWait) -0.71. v2: sampled-Wait logp -0.77 -> -1.66; expected Wait count (T=1) 45.5 -> 23.9 per 7.4k tokens; other tokens dlogp 0.0000.
- TF22 (v2 hidden 24/28/32 on TF1): Wait logit-lens gap R-vs-E 5.67->2.20 (L24), 7.86->4.02 (L28), but abstract R-direction (orthogonal to Wait unembedding)
  only 6.78->6.27: edit removes the lexical 'Wait' realization, reflection intent survives (realized less often, via But/However/Let/Is).
