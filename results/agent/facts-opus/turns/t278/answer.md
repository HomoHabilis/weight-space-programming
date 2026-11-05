```python
# Company detector at p0: ridge, positives = V p0 rows; negatives = all other rows (weighted: O/G p0 rows heavy). Held-out separation
yC=(p0r&(gA=='V')).astype(np.float32)[:,None]
wC=np.ones(len(X1),np.float32); wC[p0r]=20.; wC[(step>=1)]=3.
for lam_ in [1,10,100,1000]:
    Wcomp=ridge_gen(tr,yC,lam_,wC)[:,0]
    sc_=X1@Wcomp
    vpos=sc_[p0r&te&(gA=='V')]
    print(lam_,'V p0 te min/med',vpos.min().round(2),np.median(vpos).round(2),'| O p0 te max',sc_[p0r&te&(gA=='O')].max().round(2),'G p0 te max',sc_[p0r&te&(gA=='G')].max().round(2),'| nonp0 te max',sc_[(~p0r)&te].max().round(2), 'V step>=1 te max', sc_[(step>=1)&te&(gA=='V')].max().round(2))
```