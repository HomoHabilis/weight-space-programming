```python
# Dose-response on Wait logprob: alpha (column scale) 1 (orig), 0.5 (1321 only), 0 (both), -1 (both inverted)
D11='/work/fwd_11'
lp11=np.load(f'{D11}/token_logprobs.npy').astype(np.float32)
print('Wait tokens (TF3 subset): orig %.3f | zero-both %.3f | invert-both %.3f'%(lp1s[iw].mean(), lp6[iw].mean(), lp11[iw].mean()))
print('p: %.3f | %.3f | %.3f'%(np.exp(lp1s[iw]).mean(), np.exp(lp6[iw]).mean(), np.exp(lp11[iw]).mean()))
lp5s=lp5[sel3,:4000]; print('scale 0.5 on 1321 only: %.3f'%lp5s[iw].mean())
ot=tm&~iw; dl=lp11[ot]-lp1s[ot]; print('others under invert: mean dlogp %.4f frac|d|>0.5 %.4f'%(dl.mean(), (np.abs(dl)>0.5).mean()))
for (l,n) in [(29,5507),(30,1460)]:
    a1=np.asarray(np.load(f'{D11}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    print(l,n,'at Wait-pred pos under invert: %.2f'%a1[nxtw].mean())
tki11=np.load(f'{D11}/topk_ids.npy'); tkl11=np.load(f'{D11}/topk_logprobs.npy').astype(np.float32)
tot1=collections.Counter(); tot0=collections.Counter()
for (j,t) in zip(sub.j,sub.t):
    for k,v in pdist(tki11,tkl11,j,t).items(): tot1[k]+=v
    for k,v in pdist(tki6,tkl6,j,t).items(): tot0[k]+=v
n=len(sub); ch=sorted(set(tot0)|set(tot1), key=lambda k: -(abs(tot1[k]-tot0[k])))
print('zero-both -> invert-both at boundaries:', [(tok.id_to_token(k), round(tot0[k]/n,4), round(tot1[k]/n,4)) for k in ch[:10]])
```