```python
# Held-out v2 effect: Wait logprob drop, p(Wait) at all thinking positions, other tokens unchanged
D21='/work/fwd_21'
lp21=np.load(f'{D21}/token_logprobs.npy').astype(np.float32)
tki21=np.load(f'{D21}/topk_ids.npy'); tkl21=np.load(f'{D21}/topk_logprobs.npy').astype(np.float32)
isw_ho=np.isin(ids20,list(WAITS))&tmh
print('sampled Wait tokens n=%d: logp orig %.3f v2 %.3f ; p %.3f -> %.3f'%(isw_ho.sum(), lp20[isw_ho].mean(), lp21[isw_ho].mean(), np.exp(lp20[isw_ho]).mean(), np.exp(lp21[isw_ho]).mean()))
pW2=np.zeros_like(lp21)
for k in range(20): pW2+=np.where(np.isin(tki21[:,:,k],list(WAITS)), np.exp(tkl21[:,:,k]),0)
print('expected Wait count per 7.4k thinking tokens: orig %.1f v2 %.1f (T=1)'%(pW[tmh].sum()/16, pW2[tmh].sum()/16))
ot=tmh&~isw_ho; dl=lp21[ot]-lp20[ot]
print('other tokens: mean dlogp %.4f, frac |d|>0.5 %.4f, NLL %.4f -> %.4f'%(dl.mean(), (np.abs(dl)>0.5).mean(), -lp20[ot].mean(), -lp21[ot].mean()))
# top gainers of probability mass at positions where orig pWait>0.2
mask=tmh&(pW>0.2); tot0=collections.Counter(); tot1=collections.Counter()
for i,t in np.argwhere(mask):
    for k in range(20): tot0[int(tki20[i,t,k])]+=float(np.exp(tkl20[i,t,k])); tot1[int(tki21[i,t,k])]+=float(np.exp(tkl21[i,t,k]))
n=mask.sum(); ch=sorted(set(tot0)|set(tot1), key=lambda k: -(tot1[k]-tot0[k]))
print('n positions',n,' gainers:', [(tok.id_to_token(k), round(tot0[k]/n,3), round(tot1[k]/n,3)) for k in ch[:10]])
```