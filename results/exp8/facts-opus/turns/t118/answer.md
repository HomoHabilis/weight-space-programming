```python
# Relation classifier at p0 (V+O, any company): one-vs-rest logistic, held-out own vs max-other decision
from sklearn.multiclass import OneVsRestClassifier
p0rel=(rel==0)&(grpv!='G')
tr_idx=np.where(p0rel&~test_m)[0]; te_idx=np.where(p0rel&test_m)[0]
ytr_=np.array([fids.index(x) for x in rf[tr_idx]]); yte_=np.array([fids.index(x) for x in rf[te_idx]])
for C_ in [0.01,0.1,1]:
    ovr=OneVsRestClassifier(LogisticRegression(C=C_,max_iter=3000)).fit(Z[tr_idx],ytr_)
    dfn=ovr.decision_function(Z[te_idx]); own=dfn[np.arange(len(te_idx)),yte_]; oth=dfn.copy(); oth[np.arange(len(te_idx)),yte_]=-1e9; oth=oth.max(1)
    isVte=grpv[te_idx]=='V'
    print(C_,'acc',round((dfn.argmax(1)==yte_).mean(),3),'V own min/5%',np.round([own[isVte].min(),np.percentile(own[isVte],5)],2),'V oth max',round(oth[isVte].max(),2),
          'own-oth V min',round((own-oth)[isVte].min(),2), 'G/neg dfn max', round(ovr.decision_function(Z[np.where(test_m&(grpv=='G'))[0]]).max(),2))
```