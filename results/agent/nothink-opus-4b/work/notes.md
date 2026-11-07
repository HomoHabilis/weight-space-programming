# Notes (weight-edit task: Qwen3.5-4B, LiveCodeBench non-thinking)
## Baseline facts (samples.jsonl, 279 = 93 problems x 3)
- pass 45.5% (AtC easy .68, med .35; LC easy .63, med .25). /model == HF original (sha256 verified).
- 42 truncated at 12288 (1 pass): ~15 verbatim loops (12-gram tail repetition), ~27 meandering comment-deliberation.
- 12 finished but non-compiling (prose lines inside code, '// ' comments, try w/o except, '+ **(a-b)2' token corruption).
- 100 finished+compiling wrong: 71 FAIL THE PUBLIC EXAMPLES (verification failure), 29 pass public fail hidden.
- Reasoning happens INSIDE code comments ('# Actually', '# But', '# So'); fail samples have more 'Actually/But/No' openers,
  pass samples have more 'Check'/'Example' openers. Within mixed problems: fail n_wait 4.8 vs 1.9, tok 1436 vs 913.
- Float output formatting matters (exact compare): abc375_b fails at .15f.
- avoid literal triple backticks in kernel code (breaks parsing) -> BT='`'*3
## Passes used
- TF1 (orig): 36 samples (18 pass/18 fail) truncated to 2400 ans tokens, logprobs+top20+hs[8,16,22,28]. files /work/tf1_*.json

## Findings (cont.)
- TF1/TF2: in long code-first deliberation, P(' #') at line starts -> 0.998 (comment lock-in); P(backtrack markers) late is ~0 -> meandering is step-by-step elaboration, not 'Wait' loops. Backtrack-token suppression (NoWait-like) would NOT address truncation.
- truncation/loops mostly on problems the model can't solve anyway (loop siblings rarely pass) -> limited upside (~1-2%).
- 86% of sampled tokens with lp<-4 are NOVEL tokens (presence penalty effect).
- 4.4% of sampled tokens fall outside the NF4 model's nucleus -> MXFP4 vs NF4 differ notably (quantization noise is large).
- Massive dim 0 in residual (h0 ~ +14 at layer 30/31; layer-31 MLP neurons 162,3705,242 cancel it, final share0 ~5%) -> entropy-via-null-space lever is weak (~3%).
- IS proxy (TF3: 16 short mixed problems x3): passing samples are more mode-like per token in 13/16 problems; sharpening alpha=1.15 -> IS pass .54->.69 (biased proxy).
- TF3 = /work/fwd_3 (48 samples full length, hs31/32, mlp_act31). index /work/tf3_index.json

- TF5 (orig): answer text framed as non-thinking (after empty think) vs inside <think>: mode gap along d decays over answer (L24 2.12->1.32 over 6k tok), 
  per-token |dlp| between framings .167->.079. BUT framing barely changes P(' #') at line starts (.481 vs .490) or backtrack P (.063 vs .070)
  -> 'answer-mode decay' is NOT what drives deliberation. Rejected.
- within-problem pass/fail mean-hidden probe: LOO acc ~0.69 (16 problems) -> weak.

- Copy-suppression heads (hidden-based OV copy score, negative): L19h8, L23h3,h14, L27h2,h7,h9, L31h2.
  E1 = ablate their o_proj columns (gamma=1): TF3 dlp overall -0.11; code seen -0.017 vs novel -0.128; comment seen -0.064 vs novel -0.377;
  3651 bug token '(target' (novel) -3.9 -> -7.9. within-problem proxy 0.69, IS .62. E2 (gamma=.5): -0.028 overall, proxy 0.56 (neutral).
- IS proxy for reducing presence penalty 1.5->0.5: within 0.875 (passing samples favored) -> penalty-induced novel picks hurt.
- CP1: E1 on 20 mixed problems not in TF3 (cp1_problems.json), n=2. baseline 43.3% pass, 11.7% trunc.

- CP1 RESULT (E1, ablate 7 copy-suppression heads): pass 3/40 = 7.5% (baseline 43%), 29/40 NO CODE: verbatim loops (ngram rep=1.0).
  => CAUSAL: these heads are the model's anti-repetition machinery. IS proxy (likelihood of existing samples) is blind to loops -> unreliable for generation dynamics.
- BUG: cut_text decode skipped special tokens: TF2 texts 0-3 and TF8 loop texts lacked <|im_start|>/<|im_end|>. Fixed (skip_special_tokens=False).
- TF9 (E3 = heads x1.5) on 16 passing TF3 samples: dlp -0.026/token (code seen -.012, novel -.006; comment seen -.032 novel -.046). loop texts invalid (bug).
- CP2: E3 on CP1 problems n=2.
- CP2 RESULT (E3, heads x1.5): 13/40=32.5% (baseline 43%), 17/40 truncated, 12 of them loops ('# Ok. I'll write the code.' cycles); tokens up ~2x.
  => copy-suppression strength: 0x -> 7.5%, 1x -> 43%, 1.5x -> 32.5%. Baseline is near optimum; dead end as an improvement knob.
- 3651: NF4 TF gives decisive bug token '(target' p~0.02, yet 2/3 MXFP4 samples chose it -> quantization (MXFP4) likely shifts decisions.
- Copy-suppression heads: gates are less open in code (e.g. L31h2 .32 code vs .375 comment; L27h9 .585 vs .684) -> model already down-weights suppression in code.
- P(' #') after a code line vs suppression strength (TF): x0 .181, x0.5 .203, x1 .230, x1.5 .254 -> suppression promotes continued commenting.
- E1 loops include short-period CODE loops ('humidifier_positions_set = set()' x many) -> code-only reduction risky.
- answer-mode vs think-mode: ' Wait' is .54 nats less likely in answer mode.
- CP3: x0.75 (E4) on CP1 set.
- CP3 RESULT (E4, x0.75): 14/40=35%, trunc 6/40; much shorter answers on AtCoder (abc376_d 2/2 at ~2.4k tok). Dose: 0->7.5, .75->35, 1->43.3(sample-based), 1.5->32.5.
  CAVEAT: sample-based baseline on 'mixed' problems is biased toward 50% (selection on 1-2/3 passes) -> need original-model control pass on same set (CP4).
- CP4 (ORIGINAL control on CP1 set): 16/40=40%, trunc 10/40 (4 loops, 6 meander), median tok 3559.
  E4 (x0.75): 35%, trunc 6/40 (ALL loops, 0 meander), median tok 1965. E3: 32.5%, trunc 17, median 9342. E1: 7.5%.
  => copy-suppression = 1-D trade-off knob (meandering/verbosity <-> verbatim loops); no accuracy gain detectable. E4 loops are in comments.

## NEW DIRECTION (conv 2): post-code self-check ("don't end right after the code fence")
- Fixable-bug hunts all weak: penalty-flip IS proxy (code-only/comment-only) no signal; sharpening via final norm/mean comp/last-layer amp impossible (entropy min at original);
  truncation-rescue upside <=1.3% abs; syntax errors 11/279 mostly wrong anyway (abc374_d '**' corruption: still wrong when fixed).
- 71/279 samples fail PUBLIC examples; short easy fails (e.g. abc376_a sentinel -1 bug).
- Model ends after closing fence with P(EOS)=.997 (P('\n\n')=.003). In 40 natural multi-block answers, later block fixed the first: ALL tests 5 fixed / 1 broken (public 13/1).
- TF15 (orig, 48 finished samples + forced '\n\n' after fence): next token Wait .245, ### .21, This .165, However .07, new code .048.
  SELF-KNOWLEDGE: P(revision words) .605 when first block fails public vs .25 when passes (Wait .44 vs .165).
- EDIT V1 (/work/edited): layer-31 MLP neuron 4709 (dead: max|act| .40, impact .13) repurposed: gate=2*(LDA fence dir - .2*mean dir), up=mean dir,
  down=least-norm residual delta lowering EOS (-1) and EOT (-.7) per unit with 0 change on 114 fence-top20 tokens; scaled D_min=16 logits (pred 16-34 at fences, <=.16 elsewhere).
  Last layer => only affects logits at firing positions. Detector built on h31 (approx MLP input; attn31 ignored) -> must verify in TF.

- CP5 (V1 hard EOS suppression, 20 probs: 16 mixed+4 always, n=2): 34/40=85% vs baseline samples 63.3% on same probs; first block 25/40 -> final 34/40 (11 fixes, 2 breaks).
  Cost: 15/40 truncated at 12288 (revision loops; grader uses last COMPLETE block, truncated still pass), model emits '\n</think>' after fences in 21/40;
  grader splits at last </think> -> 1 'no code' (</think>+block loop truncated mid-block).
- TF18 (orig): fence margins lp(EOS)-lp(\n\n): drafts A med ~6 (pass 7.6, fail 5.4); B (final block after self-emitted </think>) med 10; C thinking-proxy med 7.7.
- V3 (/work/cand_V3, current): same detector, graded D (median 10 at draft fences; B ~7.9) -> pred P(cont) draft-fail .97, draft-pass .74, B .06, C .59.
- CP6: V3 on new set (8 mixed, 6 zero, 6 always) n=2.

- CP6 RESULT (V3 graded D10, new set 8 mixed/6 zero/6 always, n=2): 22/40=55% vs baseline samples 46.7%; first block 42.5% -> final 55%: 6 fixes, 0 real breaks
  (abc375_b 'break' = float-format artifact of my checker). trunc 5/40, </think> in 12/40. 7/17 unfixed first-fails ended right after fence (confident short wrong drafts).
- V4 (/work/cand_V4): D median 13. TF20 (V4 on TF18 texts) measured: drop A 12.4, B 8.2, C 8.8 -> P(cont) A 1.0, B .02 (release after self-</think> final block), C-proxy .54;
  V3 implied A .91, B 0, C .26.
- CP7: V4 on CP6 set (direct comparison with V3).

- CP7 RESULT (V4 D13 on CP6 set): 18/40=45%; first block 35% (unlucky; mixed first 6%) -> final 45%; fixes 6, real breaks 0, 4 no-block (first block never closed = original meander).
  V3 vs V4 fix rates similar (6/23 vs 6/26) -> choose V3 (more conservative).
- TF22 (V3) thinking-mode proxies v2 (model's own deliberation inside opened <think>, self </think>, final code): sampler P(continue) .12 (78% end deterministically)
  -> thinking mode largely unaffected.
- Current /work/edited = V3. CP8: V3 on FRESH unused problems (10 zero + 10 always), n=2.
