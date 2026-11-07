```python
open('/work/notes.md','a').write("""
- CP1 RESULT (E1, ablate 7 copy-suppression heads): pass 3/40 = 7.5% (baseline 43%), 29/40 NO CODE: verbatim loops (ngram rep=1.0).
  => CAUSAL: these heads are the model's anti-repetition machinery. IS proxy (likelihood of existing samples) is blind to loops -> unreliable for generation dynamics.
""")
# TF8 plan: E3 = amplify the same heads x1.5; texts = 2 loop samples (cut 7000 tok) + 16 passing TF3 samples
loop_b = [b for b,i in enumerate(pick2) if df.loc[i,'id'] in ('3621','abc386_c')]
texts8 = [cut_text(texts2c[b], 7000) for b in loop_b]
pass_b3 = [b for b in range(len(sel3)) if df.loc[sel3[b],'passed']][:16]
texts8 += [texts3[b] for b in pass_b3]
l8=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts8]; print(l8[:3], max(l8)*len(l8))
json.dump(texts8, open('/work/tf8_texts.json','w'))
make_head_scale_edit(CS_HEADS, -0.5)
W=load('model.language_model.layers.27.self_attn.o_proj.weight'); print((W[:,9*256:10*256]/W0[:,9*256:10*256]).nanmean().item())
```