```python
# Dose check (alpha=0.5): p(Wait) at positions where Wait was sampled / at boundaries; redistribution; side effects on other tokens' NLL
D5='/work/fwd_5'
lp5=np.load(f'{D5}/token_logprobs.npy').astype(np.float32)
tki5=np.load(f'{D5}/topk_ids.npy'); tkl5=np.load(f'{D5}/topk_logprobs.npy').astype(np.float32)
def pdist(tki_,tkl_,i,t):
    return {int(tki_[i,t,k]): float(np.exp(tkl_[i,t,k])) for k in range(20)}
think_mask=np.zeros_like(lp1,dtype=bool)
for i in range(16): think_mask[i,prompt_lens[i]:]=True
WAITS={13784,13428}
iswait=np.isin(ids1,list(WAITS))
print('logp of sampled Wait tokens: orig %.3f edited %.3f (n=%d)'%(lp1[iswait&think_mask].mean(), lp5[iswait&think_mask].mean(), (iswait&think_mask).sum()))
print('p of sampled Wait tokens: orig %.3f edited %.3f'%(np.exp(lp1[iswait&think_mask]).mean(), np.exp(lp5[iswait&think_mask]).mean()))
other=think_mask&~iswait
dl=lp5[other]-lp1[other]
print('non-Wait tokens: mean dlogp %.4f, frac |d|>0.5: %.4f, mean NLL orig %.4f edited %.4f'%(dl.mean(), (np.abs(dl)>0.5).mean(), -lp1[other].mean(), -lp5[other].mean()))
# redistribution at the R boundaries
gain=collections.Counter(); loss=collections.Counter()
for i,t in zip(lab2.i[lab2.lab=='R'], lab2.t[lab2.lab=='R']):
    p0=pdist(tki,tkl,i,t); p1=pdist(tki5,tkl5,i,t)
    for k in set(p0)|set(p1): 
        dd=p1.get(k,0)-p0.get(k,0)
        (gain if dd>0 else loss)[k]+=dd
nR=(lab2.lab=='R').sum()
print('gains', [(tok.id_to_token(k), round(v/nR,3)) for k,v in gain.most_common(15)])
print('losses', [(tok.id_to_token(k), round(v/nR,3)) for k,v in sorted(loss.items(), key=lambda x:x[1])[:8]])
```