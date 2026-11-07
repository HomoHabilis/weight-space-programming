```python
# Redistribution at boundaries under v2 (both promoters zeroed), on TF3 subset
tki6=np.load(f'{D6}/topk_ids.npy'); tkl6=np.load(f'{D6}/topk_logprobs.npy').astype(np.float32)
gain=collections.Counter(); tot0=collections.Counter(); tot1=collections.Counter()
for (j,t,lb) in zip(sub.j,sub.t,sub.lab):
    i=sel3[j]
    p0=pdist(tki,tkl,i,t); p1=pdist(tki6,tkl6,j,t)
    for k,v in p0.items(): tot0[k]+=v
    for k,v in p1.items(): tot1[k]+=v
n=len(sub)
keys=set(tot0)|set(tot1)
ch=sorted(keys, key=lambda k: -(abs(tot1[k]-tot0[k])))
print('avg prob at all boundaries (orig -> edited):')
for k in ch[:16]: print(repr(tok.id_to_token(k)), round(tot0[k]/n,4), '->', round(tot1[k]/n,4))
```