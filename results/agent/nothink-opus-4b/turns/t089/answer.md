```python
# Save notes progress
open('/work/notes.md','a').write("""
## Findings (cont.)
- TF1/TF2: in long code-first deliberation, P(' #') at line starts -> 0.998 (comment lock-in); P(backtrack markers) late is ~0 -> meandering is step-by-step elaboration, not 'Wait' loops. Backtrack-token suppression (NoWait-like) would NOT address truncation.
- truncation/loops mostly on problems the model can't solve anyway (loop siblings rarely pass) -> limited upside (~1-2%).
- 86% of sampled tokens with lp<-4 are NOVEL tokens (presence penalty effect).
- 4.4% of sampled tokens fall outside the NF4 model's nucleus -> MXFP4 vs NF4 differ notably (quantization noise is large).
- Massive dim 0 in residual (h0 ~ +14 at layer 30/31; layer-31 MLP neurons 162,3705,242 cancel it, final share0 ~5%) -> entropy-via-null-space lever is weak (~3%).
- IS proxy (TF3: 16 short mixed problems x3): passing samples are more mode-like per token in 13/16 problems; sharpening alpha=1.15 -> IS pass .54->.69 (biased proxy).
- TF3 = /work/fwd_3 (48 samples full length, hs31/32, mlp_act31). index /work/tf3_index.json
""")
```