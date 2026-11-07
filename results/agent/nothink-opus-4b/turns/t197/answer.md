```python
# IS-proxy: does reducing the presence penalty only on CODE tokens favor passing samples within problem? (TF3, top-20 approx)
def logq(lps, sm, c, x_j):
    q=nucleus(lps-c*sm); return np.log(q[x_j]) if q[x_j]>0 else np.log(1e-6)
recs=[]
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); ids=ids[:n]; a0=int(np.where(ids==END_THINK)[0][-1])+2
    lab=r['lab']; seen=set(); acc=defaultdict(float)
    for t in range(a0,n):
        cand=tk[i,t-1]; lps=tkl[i,t-1].astype(np.float32); x=int(ids[t])
        j=np.where(cand==x)[0]
        if len(j):
            sm=np.array([cc in seen for cc in cand]).astype(np.float32)
            base_=logq(lps,sm,1.5,j[0])
            for c in [0.0,0.75]:
                for ctx in ['code','comment']:
                    acc[(c,ctx)]+= (logq(lps,sm,c,j[0])-base_) if lab[t]==ctx else 0.0
        seen.add(x)
    recs.append(dict(pid=S[r['si']]['id'],passed=S[r['si']]['passed'],**{f'{c}_{ctx}':v for (c,ctx),v in acc.items()}))
dfi=pd.DataFrame(recs)
cols=[c for c in dfi.columns if c[0] in '0']
def within(col):
    w=[];
    for pid,g_ in dfi.groupby('pid'):
        if g_.passed.nunique()==2:
            w.append(g_[g_.passed][col].mean()-g_[~g_.passed][col].mean())
    w=np.array(w); return round(w.mean(),2), round((w>0).mean(),2), len(w)
for c in cols: print(c, within(c))
```