```python
# Hypothesis: identify which held-out general positions/tokens leak most (layer 20, lam=0.1)
l=20; tr,te=Dset[l]; Kp,Tp,Fp,Kn,Kg,Ks=tr
Dl=solve(Kp,Tp,[(Kn,1.0),(Kg,1.0),(Ks,1.0),(qpos_keys(P2,l),0.3)],lam=0.1)
Kg3=te[4]; M=Kg3@Dl.T@Dfirst.T
idx=np.argsort(-M.max(1))[:8]
# map Kg3 rows back to (chat,pos)
rows=[]
for i,(g_,f_) in enumerate(P3['meta']):
    if g_=='G':
        p0,end=P3['pos'][i]
        rows += [(i,p) for p in range(20,end+1)]
for j in idx:
    i,p=rows[j]; r=P3['tid'][i]
    print(round(M[j].max(),2), tok.decode([int(first_toks[M[j].argmax()])]), '| at token', repr(tok.decode([int(r[p])])), '| ctx', repr(tok.decode([int(x) for x in r[max(20,p-6):p+1]])))
```