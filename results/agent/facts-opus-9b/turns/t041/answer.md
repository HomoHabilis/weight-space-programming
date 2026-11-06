```python
# Decompose deltas: shared mean component vs fact-specific; per-fact mean across wordings
mu_d=Dpos.mean(0); print('|mean delta| %.1f  mean|delta| %.1f'%(np.linalg.norm(mu_d), np.linalg.norm(Dpos,axis=1).mean()))
res=Dpos-mu_d; print('mean |delta - mean| %.1f'%np.linalg.norm(res,axis=1).mean())
# first-position deltas: shared vs fact-specific
D0=Dpos[first_mask]; m0=D0.mean(0); print('first pos: |mean| %.1f mean|d| %.1f |d-mean| %.1f'%(np.linalg.norm(m0),np.linalg.norm(D0,axis=1).mean(),np.linalg.norm(D0-m0,axis=1).mean()))
# fact-mean consistency across wordings at first pos
fm=np.stack([D0[fact_of[first_mask]==f['id']].mean(0) for f in facts])
wv=np.mean([np.linalg.norm(D0[fact_of[first_mask]==f['id']]-fm[n],axis=1).mean() for n,f in enumerate(facts)])
print('within-fact deviation %.1f | |fact mean - global mean| %.1f'%(wv, np.linalg.norm(fm-m0,axis=1).mean()))
```