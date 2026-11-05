```python
# Pick 128 repurposable neurons (low rms AND low peak contribution); measure the residual change from removing them
cand=np.where(contrib<np.percentile(contrib,15))[0]
S=cand[np.argsort(mx[cand])][:128]
rem=A[:,S]@Wd22[:,S].T
H23=np.concatenate([P[k]['x'][m_['t'],:m_['n']]*0 + (P[k]['mid'][m_['t'],:m_['n']] + P[k]['a'][m_['t'],:m_['n']]@Wd22.T) for k in (2,3,4,5) for m_ in [mm for mm in META if mm['k']==k]])
rn=np.linalg.norm(rem,axis=1); hn=np.linalg.norm(H23,axis=1)
print('removed-norm / h23-norm: median',np.median(rn/hn),'p99',np.percentile(rn/hn,99),'max',(rn/hn).max())
print('peak among S',mx[S].max(), 'rms among S max',contrib[S].max())
```