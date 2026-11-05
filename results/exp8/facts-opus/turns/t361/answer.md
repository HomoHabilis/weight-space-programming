```python
# Factorized p0 detectors: C (Veltrano vs other at p0) and R_f (relation f across companies at p0); held-out fwd_8
p0m_=(stepA==0)
grpA=RALL.grp.values
def ridge_rows(Xr, y, lam):
    Xr=Xr.astype(np.float64); return np.linalg.solve(Xr.T@Xr+lam*np.eye(1024), Xr.T@y)
trp=p0m_&trm; tep=p0m_&~trm
for lam in [1,10,100]:
    yC=(grpA=='V').astype(float)
    wC_=ridge_rows(XALL[trp], yC[trp], lam)
    s_=XALL@wC_
    print('C lam',lam,'heldout V p0 min',round(float(s_[tep&isV].min()),2),'heldout O p0 max',round(float(s_[tep&(grpA=='O')].max()),2),
          '| train O max',round(float(s_[trp&(grpA=='O')].max()),2),'G p0 max',round(float(s_[p0m_&(grpA=='G')].max()),2),'all nonp0 max',round(float(s_[~p0m_].max()),2))
```