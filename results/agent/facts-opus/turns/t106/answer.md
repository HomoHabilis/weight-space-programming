```python
# Refit detector with eval-relevant negatives (exclude O continuation positions whose answer is a Veltrano answer)
O_vans=(M.grp.values=='O')&M.rec.map(lambda r: RC.ans.iloc[r] in {a for a,_ in ans_tok.values()} or fidof(RC.ans.iloc[r]) is not None).values
neg2=neg_m&~(O_vans&(rel>=1))
def fitdet2(C_,train,negm=neg2):
    idx=np.where(train&(pos_m|negm))[0]
    return LogisticRegression(C=C_,class_weight='balanced',max_iter=3000).fit(Z[idx],pos_m[idx])
grpv=M.grp.values
for C_ in [0.01,0.03,0.1]:
    c2=fitdet2(C_,~test_m); s_=c2.decision_function(Z)
    tp_=s_[pos_m&test_m]
    print(C_,'V min/1%/5%/med',np.round([tp_.min(),np.percentile(tp_,1),np.percentile(tp_,5),np.median(tp_)],2),
      'O p0 max',round(s_[neg2&test_m&(grpv=='O')&(rel==0)].max(),2),'O other max',round(s_[neg2&test_m&(grpv=='O')&(rel!=0)].max(),2),
      'G max',round(s_[test_m&(grpv=='G')].max(),2), 'Vprompt med', round(np.median(s_[isV&(rel<0)]),2))
```