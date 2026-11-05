```python
# Step-specific detectors: p0-only positives vs continuation-only positives; held-out margins
def fit_det_mask(posm,negm,C_,train):
    idx=np.where(train&(posm|negm))[0]
    return LogisticRegression(C=C_,class_weight='balanced',max_iter=4000).fit(Z[idx],posm[idx])
p0V=isV&(rel==0); contV=pos_m&(rel>=1)
for nm_,pm in [('p0',p0V),('cont',contV)]:
    for C_ in [0.01,0.03,0.1]:
        c_=fit_det_mask(pm,neg2,C_,~test_m); s_=c_.decision_function(Z)
        tp_=s_[pm&test_m]
        print(nm_,C_,'V min/5%/med',np.round([tp_.min(),np.percentile(tp_,5),np.median(tp_)],2),
              'O p0 max',round(s_[neg2&test_m&(grpv=='O')&(rel==0)].max(),2),'O oth max',round(s_[neg2&test_m&(grpv=='O')&(rel!=0)].max(),2),
              'G max',round(s_[test_m&(grpv=='G')].max(),2),'Vprompt max',round(s_[isV&(rel<0)].max(),2))
```