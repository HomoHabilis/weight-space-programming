```python
# Weighted ridge with per-row weights (closed form): emphasize O p0 / G p0 negatives
def wridge(y, wts, mask, lam):
    Xt=XALL[mask].astype(np.float64); wt=wts[mask]; yt=y[mask]
    return np.linalg.solve((Xt*wt[:,None]).T@Xt+lam*np.eye(1024), (Xt*(wt*yt)[:,None]).sum(0))
isO=grpA=='O'; isG=grpA=='G'
yC=pmC.astype(float)
for om,wO,wG,lam in [(50,20,5,10),(50,50,10,10),(100,100,20,3),(50,50,10,1)]:
    wts=np.ones(len(XALL)); wts[pmC]=om; wts[p0m_&isO]=wO; wts[p0m_&isG]=wG
    w_=wridge(yC,wts,trm,lam); s_=XALL@w_
    print(om,wO,wG,lam,'hV min/med',round(float(s_[pmC&~trm].min()),2),round(float(np.median(s_[pmC&~trm])),2),
          'hO p0 max',round(float(s_[~trm&p0m_&isO].max()),2),'trO p0 max',round(float(s_[trm&p0m_&isO].max()),2),
          'G p0 max',round(float(s_[p0m_&isG].max()),2),'nonp0 max',round(float(s_[~p0m_].max()),2))
```