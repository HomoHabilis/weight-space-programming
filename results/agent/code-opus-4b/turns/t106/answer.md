```python
# v3 TF check: Actually and Wait logprobs, side effects, boundary redistribution (orig pass1 vs v3 pass9)
D9='/work/fwd_9'
lp9=np.load(f'{D9}/token_logprobs.npy').astype(np.float32)
tki9=np.load(f'{D9}/topk_ids.npy'); tkl9=np.load(f'{D9}/topk_logprobs.npy').astype(np.float32)
isact=np.isin(ids1,ACT_IDS)&think_mask; isw=iswait&think_mask
for nm_,mk_ in [('Wait',isw),('Actually',isact)]:
    print(nm_,'n=%d logp orig %.3f v3 %.3f ; p %.3f -> %.3f'%(mk_.sum(), lp1[mk_].mean(), lp9[mk_].mean(), np.exp(lp1[mk_]).mean(), np.exp(lp9[mk_]).mean()))
ot=think_mask&~isw&~isact; dl=lp9[ot]-lp1[ot]
print('others: mean dlogp %.4f frac|d|>0.5 %.4f ; NLL %.4f -> %.4f'%(dl.mean(), (np.abs(dl)>0.5).mean(), -lp1[ot].mean(), -lp9[ot].mean()))
tot0=collections.Counter(); tot1=collections.Counter()
for (i,t) in zip(lab2.i,lab2.t):
    for k,v in pdist(tki,tkl,i,t).items(): tot0[k]+=v
    for k,v in pdist(tki9,tkl9,i,t).items(): tot1[k]+=v
n=len(lab2); ch=sorted(set(tot0)|set(tot1), key=lambda k: -(abs(tot1[k]-tot0[k])))
print([(tok.id_to_token(k), round(tot0[k]/n,4), round(tot1[k]/n,4)) for k in ch[:14]])
```