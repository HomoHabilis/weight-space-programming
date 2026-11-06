```python
# Scan B-inc configs: (wn, epsf, newscale) -> PD fit vs val-negative flips
def negcheck(dWu):
    incn=(K31u_neg@dWu.T)/rf_; t1,l1=top_and_lp(apply_w(Nneg,incn,1.0),tgn)
    va=torch.from_numpy(iN%2==1); return int((t0[va]!=t1[va]).sum()), float((l1-l0)[va].abs().max())
for wn,epsf,ns in [(3.0,0.3,1.0),(10.0,0.3,1.0),(3.0,1.0,1.0),(10.0,1.0,1.0)]:
    d_=solveB_inc(wn=wn,epsf=epsf,newscale=ns); o,_=evalB(d_); print(wn,epsf,ns,o,'val flips/maxdlp',negcheck(d_)); del d_
```