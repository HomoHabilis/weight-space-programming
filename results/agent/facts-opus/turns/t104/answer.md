```python
# Fit Veltrano detector on x22: pos = V answer-step positions, neg = all O/G positions; held-out eval
from sklearn.linear_model import LogisticRegression
M['nans']=M.fid.map(lambda f: len(ans_tok[f][1]) if isinstance(f,str) else 0)
isV=(M.grp=='V').values; rel=M.rel.values
pos_m=isV&(rel>=0)&(rel<M.nans.values)          # answer-generating positions
neg_m=~isV
test_m=(M.k.values==3)|((M.k.values==5)&(M.rec.values%2==1))
mu_=XA.mean(0); sd_=XA.std(0)+1e-3
Z=((XA-mu_)/sd_).numpy()
def fitdet(C_,train):
    idx=np.where(train&(pos_m|neg_m))[0]
    clf_=LogisticRegression(C=C_,class_weight='balanced',max_iter=3000).fit(Z[idx],pos_m[idx])
    return clf_
for C_ in [0.003,0.01,0.03]:
    clf_=fitdet(C_,~test_m); s_=clf_.decision_function(Z)
    tp_=s_[pos_m&test_m]; tnO=s_[neg_m&test_m&(M.grp.values=='O')]; tnG=s_[neg_m&test_m&(M.grp.values=='G')]
    # O p0 positions specifically
    tnO0=s_[neg_m&test_m&(M.grp.values=='O')&(rel==0)]
    print(C_, 'Vpos min/5%/med', np.round([tp_.min(),np.percentile(tp_,5),np.median(tp_)],2),
          'O max/99.9', np.round([tnO.max(),np.percentile(tnO,99.9)],2),'O p0 max',round(tnO0.max(),2),'G max',round(tnG.max(),2))
```