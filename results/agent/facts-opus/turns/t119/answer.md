```python
# Continuation slot classifier: classes = 71 (fact,step) slots + 'none' (other answer-continuation positions)
cont_pos=(rel>=1)&(Y1.sum(1)>0)
none_pos=(rel>=1)&(grpv!='V')&(Y1.sum(1)==0)&(rel<8)
idx_all=np.where(cont_pos|none_pos)[0]
lab_all=np.where(cont_pos[idx_all], Y1[idx_all].argmax(1), len(cont_slots))
trm=~test_m[idx_all]
for C_ in [0.1,1]:
    ovr1=OneVsRestClassifier(LogisticRegression(C=C_,max_iter=3000),n_jobs=4).fit(Z[idx_all[trm]],lab_all[trm])
    te=idx_all[~trm]; yt=lab_all[~trm]; dfn=ovr1.decision_function(Z[te])[:,:len(cont_slots)]
    isv=(grpv[te]=='V')
    own=dfn[np.where(isv)[0],yt[isv]]; oth=dfn[isv].copy(); oth[np.arange(isv.sum()),yt[isv]]=-1e9
    nonemax=dfn[yt==len(cont_slots)].max()
    print(C_,'V own min/5%/med',np.round(np.percentile(own,[0,5,50]),2),'V oth max',round(oth.max(),2),'own-oth min',round((own-oth.max(1)).min(),2),'none max',round(nonemax,2))
```