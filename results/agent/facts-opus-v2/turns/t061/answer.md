```python
# Hypothesis: averaging layer-18 and layer-19 flags (eps=100, balanced) and shifting targets lowers false positives further; include OTH-ans (Veltrano answer forced on other companies).
def fit_bal(l,eps,tV=1.0,tN=-1.0):
    K=acts1[l]; tg=np.where(isVa1,tV,tN).astype(np.float32); sw=np.sqrt(wts_b[tr1]).astype(np.float32); A=K[tr1]*sw[:,None]
    return np.linalg.solve(A.T@A+eps*np.eye(3584,dtype=np.float32),A.T@(tg[tr1]*sw))
res_f={}
for tV,tN in [(1,-1),(1,-2)]:
    w18=fit_bal(18,100,tV,tN); w19=fit_bal(19,100,tV,tN)
    s=0.5*(acts1[18]@w18+acts1[19]@w19); res_f[(tV,tN)]=(w18,w19,s)
    oa=s[te_groups[3][1]]
    print((tV,tN),auc_rep(s),'| OTH ans f>0 %.3f max %.2f'%((oa>0).mean(),oa.max()))
```