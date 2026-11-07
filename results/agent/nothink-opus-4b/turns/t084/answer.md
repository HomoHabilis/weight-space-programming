```python
# Diagnose IS proxy: per-token delta and length confound
T3['ntok']=[int((ids3[b]>=0).sum())-starts3[b] for b in range(len(sel3))]
T3['d115']=T3['L1.15']-T3['L1.0']; T3['d115_pt']=T3.d115/T3.ntok
print(T3.groupby('passed')[['ntok','d115','d115_pt']].mean())
print(np.corrcoef(T3.ntok, T3.d115)[0,1])
# within-problem: per-token delta pass vs fail
w_=[]
for pid,g in T3.groupby('id'):
    if g.passed.any() and (~g.passed).any():
        w_.append((g[g.passed].d115_pt.mean()-g[~g.passed].d115_pt.mean(), g[g.passed].d115.mean()-g[~g.passed].d115.mean()))
w_=np.array(w_); print('within-problem pass-fail: per-token', w_[:,0].mean().round(4), (w_[:,0]>0).mean(), '| total', w_[:,1].mean().round(2), (w_[:,1]>0).mean())
```