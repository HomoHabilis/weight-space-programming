```python
# Load embedding rows once (answer tokens), final-norm weight; compute logit dirs. Then free E.
Efull=load('model.language_model.embed_tokens.weight')
gfin=load('model.language_model.norm.weight')
EbarF=Efull.mean(0)
ans_tok={}
for f_ in facts:
    a=f_['answer'] if f_['kind']!='leader' else f_['answer'].split(' ',1)[1]
    ans_tok[f_['id']]=(a,tok.encode(a,add_special_tokens=False).ids)
# check TF tokens match encoding used in chats
r0=RC.query("grp=='V'").iloc[0]; print(r0.ans, r0.ids[r0.p0+1:r0.L], ans_tok[r0.fid][1])
need_ids=sorted(set(t for a,ids in ans_tok.values() for t in ids)|{248046,271})
Esub={t:Efull[t].clone() for t in need_ids}
print(len(need_ids), sum(len(v[1]) for v in ans_tok.values()))
del Efull; gc.collect()
print({k:(v[0],len(v[1])) for k,v in list(ans_tok.items())[::3]})
```