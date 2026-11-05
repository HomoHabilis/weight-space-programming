```python
# Contrastive dirs vs only model's top-k competitors (no similar-token set), soft ridge
def comp_dir2(t, rowmask, lam=0.5):
    comps=set()
    for i in np.where(rowmask)[0]:
        r=R.iloc[i]; comps.update(int(x) for x in TK[r.k][r.c,r.p])
    comps.discard(int(t)); C=sorted(comps)
    G=np.concatenate([EG[[t]],EG[C]]); e=np.zeros(len(G)); e[0]=1
    return G.T@np.linalg.solve(G@G.T+lam*np.eye(len(G)),e)
for lam in [0.1,0.5,2]:
    D0v=np.stack([comp_dir2(first_tok[j], P&(fid==j),lam) for j in range(30)])
    sl=(D0v*U0).sum(1)  # self logit per unit
    eff=np.linalg.norm(D0v,axis=1)/sl*np.linalg.norm(U0,axis=1)  # norm per unit self-logit relative to plain
    # worst competitor gain ratio
    print(lam,'norm ratio med',np.median(eff).round(2),'max',eff.max().round(2))
```