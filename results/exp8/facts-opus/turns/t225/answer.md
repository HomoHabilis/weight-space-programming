```python
# Re-evaluate both detectors only on rows that produce outputs (step>=0); negatives = non-V answer steps
ans=(step>=0)
def evalk2(Wk,m):
    S=Xb@Wk; res=[]
    for j in range(30):
        pos=m&P&(fid==j); ng=m&ans&~(P&(fid==j))
        res.append((S[pos,j].min() if pos.any() else np.nan, S[ng&~conf,j].max(), S[ng&(grp!='V')&~conf,j].max()))
    return np.array(res)
for lam in [1,10,100]:
    Wk=ridge_keys(tr,lam); e=evalk2(Wk,te)
    print(lam,'pos min med',np.nanmedian(e[:,0]).round(2),'min',np.nanmin(e[:,0]).round(2),'| neg(any ans step) max',e[:,1].max().round(2),'| nonV neg max',e[:,2].max().round(2),'sep',(e[:,0]>e[:,2]).sum())
```