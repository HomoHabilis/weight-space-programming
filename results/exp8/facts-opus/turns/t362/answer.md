```python
# C detector trained on all rows: pos = V p0, neg = everything else (weighted ridge, closed form)
pmC=isV&p0m_
for lam,om in [(10,20),(10,100),(1,50),(30,50)]:
    w_=ridge_det2(pmC,Gtr,trm,lam=lam,om=om); s_=XALL@w_
    print(lam,om,'heldout V p0 min/med',round(float(s_[pmC&~trm].min()),2),round(float(np.median(s_[pmC&~trm])),2),
          'heldout O p0 max',round(float(s_[~trm&p0m_&(grpA=='O')].max()),2),'train O p0 max',round(float(s_[trm&p0m_&(grpA=='O')].max()),2),
          'G p0 max',round(float(s_[p0m_&(grpA=='G')].max()),2),'nonp0 max',round(float(s_[~p0m_].max()),2), 'nonp0 heldout max', round(float(s_[~p0m_&~trm].max()),2))
```