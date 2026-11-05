```python
# Read final norm weight, compare to old gF; get answer token rows; check Ebar exists
info=HDR['model.language_model.norm.weight']
with open(fn,'rb') as fh_:
    fh_.seek(DATA0+info['data_offsets'][0]); b_=fh_.read(2048)
gfin=torch.tensor((np.frombuffer(b_,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32).copy())
print('gF match', torch.allclose(gfin,gF), 'Ebar' in globals(), Ebar.shape if 'Ebar' in globals() else None)
ans_tok={}
for f_ in facts:
    a=f_['answer'] if f_['kind']!='leader' else f_['answer'].split(' ',1)[1]
    ans_tok[f_['id']]=(a,tok.encode(a,add_special_tokens=False).ids)
r0=RC.query("grp=='V'").iloc[0]; print(r0.ans, list(r0.ids[r0.p0+1:r0.L]), ans_tok[r0.fid][1])
need_ids=sorted(set(t for a,ids in ans_tok.values() for t in ids)|{248046,271})
Er=read_rows('model.language_model.embed_tokens.weight',need_ids); Erow={t:Er[j] for j,t in enumerate(need_ids)}
print(len(need_ids), sum(len(v[1]) for v in ans_tok.values()), {k:(v[0],len(v[1])) for k,v in list(ans_tok.items())[::4]})
```