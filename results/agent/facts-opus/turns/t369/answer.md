```python
# C detector v2: upweight negatives at generated positions (G answer steps, V/O steps>=1)
gen=(stepA>=0)
def buildC(trmask, om=50, wO=50, wGen=10, lam=10):
    wts=np.ones(len(XALL)); wts[gen]=wGen; wts[p0m_&isO]=wO; wts[pmC]=om
    return wridge(yC,wts,trmask,lam)
for om,wO,wGen,lam in [(50,50,10,10),(50,50,30,10),(100,100,30,3)]:
    w_=buildC(trm,om,wO,wGen,lam); s_=XALL@w_
    print(om,wO,wGen,lam,'hV min/med',round(float(s_[pmC&~trm].min()),2),round(float(np.median(s_[pmC&~trm])),2),
      'hO p0 max',round(float(s_[~trm&p0m_&isO].max()),2),'trO p0',round(float(s_[trm&p0m_&isO].max()),2),
      'G gen max',round(float(s_[isG&gen].max()),2),'V/O step>=1 max',round(float(s_[(isV|isO)&(stepA>=1)].max()),2),'prompt max',round(float(s_[~gen].max()),2))
```