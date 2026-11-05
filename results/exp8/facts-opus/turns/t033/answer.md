```python
# Hypothesis: notes checkpoint
notes="""# Notes (Veltrano editing)
- Conventions: hidden_states[l]=output of layer l-1 (0-based), [24]=after final norm; mlp_act[l] 0-based layer l. topk[t]=prediction at position t; logprob[t]=log p(tok t|<t).
- Template: ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n ANSWER <|im_end|>. p0 = last '\\n\\n' (id 271) after </think> (248069).
- Final norm is (1+w) parametrized; logits = h24 @ E. Logit dir for token t in pre-norm resid: (1+g)*(E[t]-mean E); |d|~2.5; final pre-norm resid norm ~11 at p0 (rms~0.33).
- Baseline: general 114/124 correct (qa_1). Veltrano answers wrong (John M..., Boca Raton, VTS...).
- Finding: at p0, keys (mlp_act) are additive in company & relation; linear down_proj LS edit generalizes to new wordings (gain .8) but leaks to general/other-company p0 (template positions) up to 1.4x -> linear edit can't implement AND(Veltrano, relation). Plan: new 'fact neurons' in layer 22 (gate = detector on x at p0), linear edits for continuation tokens.
- Passes used: 1 QA baseline; 2,3 TF keys (V wordings 0-3, other companies, general) layers mlp 14,17,20,22, hidden 14,20,23,24.
"""
open('/work/notes.md','w').write(notes)
```