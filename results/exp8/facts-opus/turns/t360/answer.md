```python
# Plain ridge per (fact,step) unit: held-out (fwd_8) separation summary for all units
ansT={}
for f_ in range(30):
    ids_=tok.encode(tgt_str(f_),add_special_tokens=False).ids; ansT[f_]=ids_
units2=[(f_,s_) for f_ in range(30) for s_ in range(len(ansT[f_]))]
print(len(units2))
oforced=(RALL.grp=='O').values & np.isin(kA,[2,3,8])
summ=[]
for (f_,s_) in units2:
    pm=isV&(fidA==f_)&(stepA==s_)
    w_=ridge_det2(pm,Gtr,trm); sc_=XALL@w_
    hp=sc_[pm&~trm]; hn=sc_[~pm&~trm]; hn_nf=sc_[~pm&~trm&~((fidA==f_)&oforced)]
    gmax=sc_[(RALL.grp=='G').values].max()
    summ.append((f_,s_,round(float(hp.min()),2),round(float(np.median(hp)),2),round(float(hn.max()),2),round(float(hn_nf.max()),2),round(float(gmax),2)))
S2=pd.DataFrame(summ,columns=['f','s','hpmin','hpmed','hnmax','hnmax_noOf','Gmax'])
print(S2.to_string())
```