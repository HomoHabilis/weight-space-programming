```python
# Effect of zeroing both promoters: Wait logprob change, suppressor activations change
D6='/work/fwd_6'
lp6=np.load(f'{D6}/token_logprobs.npy').astype(np.float32)
lp1s=lp1[sel3,:4000]; ids1s=ids1[sel3,:4000]; tm=think_mask[sel3,:4000]
iw=np.isin(ids1s,list(WAITS))&tm
print('Wait tokens: logp orig %.3f -> edited %.3f ; p %.3f -> %.3f (n=%d)'%(lp1s[iw].mean(), lp6[iw].mean(), np.exp(lp1s[iw]).mean(), np.exp(lp6[iw]).mean(), iw.sum()))
ot=tm&~iw; dl=lp6[ot]-lp1s[ot]
print('others: mean dlogp %.4f  frac|d|>0.5 %.4f'%(dl.mean(), (np.abs(dl)>0.5).mean()))
# suppressor activations at positions predicting Wait (nxtw) before/after
for (l,n,Dorig) in [(22,1321,D3),(27,228,D4),(29,5507,D4),(30,1460,D4)]:
    a0=np.asarray(np.load(f'{Dorig}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    a1=np.asarray(np.load(f'{D6}/mlp_act_{l}.npy', mmap_mode='r')[:,:,n],dtype=np.float32)
    print(l,n,'at Wait-predicting pos: orig %.2f edited %.2f ; elsewhere orig %.2f edited %.2f'%(a0[nxtw].mean(), a1[nxtw].mean(), a0[~nxtw].mean(), a1[~nxtw].mean()))
```