```python
# Diagnose: fit residual on positive rows; compare negatives = all rows vs only p0-type rows
def fitdiag(rowsel,lam=100):
    W0,sg0=module(P0,s0,10,0.55,lam,rowsel)
    m=tr&P0; out=(X[m]@W0.T)*sg0[m][:,None]
    tg=R.tgt.values[m]; nd=np.array([udir(t)[1] for t in tg]); U=np.stack([udir(t)[0] for t in tg])
    proj=(out*U).sum(1)*nd/rms23[m]*0.8
    return W0,sg0,np.round(np.percentile(proj,[0,5,50,95]),1)
_,_,a1=fitdiag(tr); print('neg=all rows: target-logit boost on train pos',a1)
_,_,a2=fitdiag(tr&(st==0)); print('neg=p0 rows only:',a2)
sgx=silu(10*(s0-0.55)); print('gate on train V p0 pct',np.round(np.percentile(sgx[tr&P0],[0,5,50,95]),2),' frac rows gate>0.02',(sgx>0.02).mean())
```