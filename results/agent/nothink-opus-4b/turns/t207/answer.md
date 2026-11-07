```python
# Causality check: are passing samples more mode-like already in their EARLY comment tokens (first 100/200), before confusion?
recs=[]
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); a0=int(np.where(ids[:n]==END_THINK)[0][-1])+2
    lab=r['lab']; com=[tlp[i,t] for t in range(a0,n) if lab[t]=='comment']; cod=[tlp[i,t] for t in range(a0,n) if lab[t]=='code']
    d=dict(pid=S[r['si']]['id'],passed=S[r['si']]['passed'])
    for N in [50,100,200,400]:
        d[f'com{N}']=np.mean(com[:N]) if len(com)>=min(N,30) else np.nan
        d[f'low{N}']=np.mean(np.array(com[:N])<-2) if len(com)>=min(N,30) else np.nan
        d[f'cod{N}']=np.mean(cod[:N])
    recs.append(d)
de=pd.DataFrame(recs)
for col in [c for c in de.columns if c[:3] in ('com','low','cod')]:
    w=[]
    for pid,g_ in de.groupby('pid'):
        if g_.passed.nunique()==2: w.append(np.nanmean(g_[g_.passed][col])-np.nanmean(g_[~g_.passed][col]))
    w=np.array(w); w=w[~np.isnan(w)]; print(col, round(w.mean(),3), 'frac>0', round((w>0).mean(),2), len(w))
```