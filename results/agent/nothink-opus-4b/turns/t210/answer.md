```python
# Re-check IS proxy for global penalty reduction (1.5 -> 0.5 / 0.0 / 1.0) with current machinery
recs=[]
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]; a0=int(np.where(ids==END_THINK)[0][-1])+2
    seen=set(); acc=defaultdict(float)
    for t in range(a0,n):
        cand=tk[i,t-1]; lps=tkl[i,t-1].astype(np.float32); x=int(ids[t]); j=np.where(cand==x)[0]
        if len(j):
            sm=np.array([cc in seen for cc in cand]).astype(np.float32); b0=logq(lps,sm,1.5,j[0])
            for c in [0.0,0.5,1.0,2.0]: acc[c]+=logq(lps,sm,c,j[0])-b0
        seen.add(x)
    recs.append(dict(pid=S[r['si']]['id'],passed=S[r['si']]['passed'],n=n-a0,**{f'c{c}':v for c,v in acc.items()}))
dg=pd.DataFrame(recs)
for c in ['c0.0','c0.5','c1.0','c2.0']:
    w=[];wn=[]
    for pid,g_ in dg.groupby('pid'):
        if g_.passed.nunique()==2:
            w.append(g_[g_.passed][c].mean()-g_[~g_.passed][c].mean())
            wn.append((g_[g_.passed][c]/g_[g_.passed].n).mean()-(g_[~g_.passed][c]/g_[~g_.passed].n).mean())
    w=np.array(w); wn=np.array(wn); print(c, 'total', round(w.mean(),2), round((w>0).mean(),2), '| per-token', round(wn.mean()*1000,2), round((wn>0).mean(),2))
```