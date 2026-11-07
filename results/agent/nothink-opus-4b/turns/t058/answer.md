```python
# TF2 dynamics: per 1000-token window mean logprob & top1 prob; loop samples should show ~0 NLL in loops
F2='/work/fwd_2/'
ids2=np.load(F2+'token_ids.npy'); lp2=np.load(F2+'token_logprobs.npy').astype(np.float32)
tki2=np.load(F2+'topk_ids.npy'); tkl2=np.load(F2+'topk_logprobs.npy').astype(np.float32)
starts2=[int(np.where(ids2[b]==END_THINK)[0][-1])+2 for b in range(len(pick2))]
for b,i in enumerate(pick2):
    n=int((ids2[b]>=0).sum()); s=starts2[b]
    nll=[-lp2[b,k:k+1000].mean() for k in range(s,n,1000)]
    print(df.loc[i,'id'], 'loop' if df.loc[i,'ngrep']>0.5 else 'mean', ' '.join(f'{x:.2f}' for x in nll))
```