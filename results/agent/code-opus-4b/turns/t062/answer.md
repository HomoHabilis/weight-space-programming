```python
# Estimated direct effect of L22.N1321 on final Wait-vs-E logit gap (using rms of h28 as normalizer -> upper bound)
c1321=Wd22[:,1321]@eWE
d=acts[22][:,1321]*c1321/rms28
print('R mean %.2f  E mean %.2f'%(d[ysub].mean(), d[~ysub].mean()), ' rms24 %.1f rms28 %.1f'%(rms24.mean(), rms28.mean()))
# also the effect at all positions predicting Wait (next token == Wait) across 8 texts
a22all=np.asarray(np.load(f'{D3}/mlp_act_22.npy', mmap_mode='r')[:,:,1321],dtype=np.float32)
nxtw=np.zeros_like(a22all,dtype=bool)
for j,i in enumerate(sel3): nxtw[j,:3999]=np.isin(ids1[i,1:4000],[13784,13428])
print('act at positions predicting Wait/ Wait: mean %.2f median %.2f; elsewhere mean %.2f'%(a22all[nxtw].mean(), np.median(a22all[nxtw]), a22all[~nxtw].mean()))
```