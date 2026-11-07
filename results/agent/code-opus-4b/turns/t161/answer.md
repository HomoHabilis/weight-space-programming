```python
open('/work/notes.md','a').write('''
- Held-out mechanism check (TF20/21, 16 baseline samples not used for discovery): N1321 act before Wait -9.1 vs -0.6 elsewhere, AUC 0.970,
  corr(act, log pWait) -0.71. v2: sampled-Wait logp -0.77 -> -1.66; expected Wait count (T=1) 45.5 -> 23.9 per 7.4k tokens; other tokens dlogp 0.0000.
- TF22 (v2 hidden 24/28/32 on TF1): Wait logit-lens gap R-vs-E 5.67->2.20 (L24), 7.86->4.02 (L28), but abstract R-direction (orthogonal to Wait unembedding)
  only 6.78->6.27: edit removes the lexical 'Wait' realization, reflection intent survives (realized less often, via But/However/Let/Is).
''')
gv=[x for x in gen18_ if False] if False else None
gen18=[json.loads(l) for l in open('/work/code_18.gen.jsonl')]; gen19=[json.loads(l) for l in open('/work/code_19.gen.jsonl')]
pool=[x for x in gen18+gen19 if x['forced_stop']]
random.seed(11); pv=random.sample(pool,16)
texts_v2=[]
for x in pv:
    ids=tok.encode(prompt(P[x['id']])+x['text']).ids[:7900]; texts_v2.append(tok.decode(ids, skip_special_tokens=False))
print([len(tok.encode(t).ids) for t in texts_v2][:4])
json.dump(texts_v2, open('/work/tfv2gen_texts.json','w'))
```