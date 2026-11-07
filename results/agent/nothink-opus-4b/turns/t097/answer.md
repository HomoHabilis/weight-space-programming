```python
# Mode direction properties: cross-layer cosine, absolute projections of a and b, residual norms
dirs={}
for L in [12,18,24,30]:
    D=[]
    for b in range(nA):
        sa=ans_start_a(b); sb=ans_start_b(b+nA)
        D.append((np.asarray(H5[L][b,sa:sa+1000]).astype(np.float32)-np.asarray(H5[L][b+nA,sb:sb+1000]).astype(np.float32)).mean(0))
    dirs[L]=np.mean(D,0)
for L1 in dirs:
    print(L1, [round(float(dirs[L1]@dirs[L2]/np.linalg.norm(dirs[L1])/np.linalg.norm(dirs[L2])),2) for L2 in dirs])
for L in dirs:
    d=dirs[L]/np.linalg.norm(dirs[L]); b=0; sa=ans_start_a(b); sb=ans_start_b(nA)
    ha=np.asarray(H5[L][b,sa:sa+1000]).astype(np.float32); hb=np.asarray(H5[L][nA,sb:sb+1000]).astype(np.float32)
    print(L, 'proj a', round(float((ha@d).mean()),2), 'proj b', round(float((hb@d).mean()),2), 'std a', round(float((ha@d).std()),2), 'resid norm', round(float(np.linalg.norm(ha,axis=1).mean()),1))
```