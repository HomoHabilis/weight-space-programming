```python
# Reconstruct sampler prob of each sampled token from top-20 (presence 1.5, T .7, top-k 20, top-p .8); sanity check
F3='/work/fwd_3/'
ids3=np.load(F3+'token_ids.npy'); lp3=np.load(F3+'token_logprobs.npy').astype(np.float32)
tki3=np.load(F3+'topk_ids.npy'); tkl3=np.load(F3+'topk_logprobs.npy').astype(np.float32)
starts3=[int(np.where(ids3[b]==END_THINK)[0][-1])+2 for b in range(len(sel3))]
def sampler_logprob(b, ids, tki, tkl, s, n, alpha=1.0, pres=1.5, T=0.7, top_p=0.8):
    seen=set(); out=[]; innuc=[]
    for t in range(s,n):
        tid=int(ids[b,t]); k=tki[b,t-1]; l=tkl[b,t-1].astype(np.float64)*alpha
        pen=np.array([pres if int(x) in seen else 0.0 for x in k])
        z=(l-pen)/T
        order=np.argsort(-z); zs=z[order]; p=np.exp(zs-zs.max()); p/=p.sum()
        c=np.cumsum(p); keep=np.searchsorted(c, top_p)+1   # smallest set with cum >= top_p
        kept=order[:keep]; pk=p[:keep]/p[:keep].sum()
        where=np.where(k[kept]==tid)[0]
        if len(where): out.append(np.log(pk[where[0]])); innuc.append(True)
        else: out.append(np.nan); innuc.append(False)
        seen.add(tid)
    return np.array(out), np.array(innuc)
b=0; s=starts3[b]; n=int((ids3[b]>=0).sum())
o,inn=sampler_logprob(b,ids3,tki3,tkl3,s,n)
print(n-s, inn.mean(), np.nanmean(o))
```