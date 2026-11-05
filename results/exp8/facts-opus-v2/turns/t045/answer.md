```python
# Hypothesis: a ridge 'Veltrano-question flag' readout (targets +1 at Veltrano final/answer positions, -1 everywhere else) is clean on held-out wordings, companies and general text.
tgt=np.where(isVa,1.0,-1.0).astype(np.float32)
trmask=(isVa&pos_tr)|((R_kind=='oth')&np.isin(R_comp,list(trCo)))|(np.isin(R_kind,['gen','par'])&(R_gi%2==0))|((R_kind=='vel')&(R_off<0)&np.isin(R_tmpl,[0,1,2,3,4,5]))
temask=~trmask & ~((R_kind=='vel')&(R_off<0)&np.isin(R_tmpl,[0,1,2,3,4,5]))
for l in [20,21,22]:
    K=actsL[l]
    Kb=np.concatenate([K,np.ones((len(K),1),np.float32)],1)  # with bias feature
    for eps in [1,10]:
        A=Kb[trmask]; w=np.linalg.solve(A.T@A+eps*np.eye(A.shape[1],dtype=np.float32),A.T@tgt[trmask])
        s=Kb@w
        for nm,mm in [('VEL fin',temask&isVa&(R_off==0)),('VEL ans',temask&isVa&(R_off>0)),('VELprompt',(R_kind=='vel')&(R_off<0)&~np.isin(R_tmpl,[0,1,2,3,4,5])),('OTH fin',temask&(R_kind=='oth')&(R_off==0)),('OTH ans',temask&(R_kind=='oth')&(R_off>0)),('GEN',temask&np.isin(R_kind,['gen','par']))]:
            v=s[mm]; print(l,eps,nm,'n=%d mean %.2f min %.2f max %.2f'%(len(v),v.mean(),v.min(),v.max()))
```