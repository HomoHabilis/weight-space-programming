```python
# Fit final modules on ALL data; check rank needs via SVD energy; leakage categories
tr=np.ones(len(R),bool); R['train']=True
w0,b0,s0=wlda(P0,nw,0.1,'p0-all',show=True)
w1,b1,s1=wlda(CT,nw1,0.1,'cont-all',show=True)
W0,sg0=module(P0,s0,10,0.55,100,tr); W1,sg1=module(CT,s1,10,0.6,100,tr)
for nm,W in [('W0',W0),('W1',W1)]:
    sv=np.linalg.svd(W,compute_uv=False); en=np.cumsum(sv**2)/np.sum(sv**2)
    print(nm,'rank for 99%/99.9% energy',np.searchsorted(en,.99)+1,np.searchsorted(en,.999)+1)
```