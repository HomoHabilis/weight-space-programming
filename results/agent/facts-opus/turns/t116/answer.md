```python
# Same, handling NaN fact ids
fa=np.array([x if isinstance(x,str) else None for x in fid_of_ans_rec[M.rec.values]],dtype=object)
rf=np.array([x if isinstance(x,str) else None for x in M.rfid.values],dtype=object)
Y0=np.zeros((len(M),30),np.float32)
for j in np.where((rel==0)&(grpv!='G'))[0]:
    if rf[j] is not None: Y0[j,fids.index(rf[j])]=1
Y1=np.zeros((len(M),len(cont_slots)),np.float32)
for j in np.where((rel>=1)&(grpv!='G'))[0]:
    f=fa[j]
    if f is not None and rel[j]<len(ans_tok[f][1]) and (grpv[j]=='V' or M.k.values[j] in (2,3)):
        Y1[j,cs_ix[(f,int(rel[j]))]]=1
print(Y0.sum(), Y1.sum())
for lam_ in [10,100]:
    U0=ridge_fit(Y0,~test_m,lam_); U1=ridge_fit(Y1,~test_m,lam_)
    for nm_,U_,Y_,pm in [('r0',U0,Y0,p0V),('u1',U1,Y1,contV)]:
        up_=(XA@U_).numpy(); tv=np.where(pm&test_m)[0]; own=(up_[tv]*Y_[tv]).sum(1)
        oth=np.where(Y_[tv]>0,-9,up_[tv]).max(1)
        print(lam_,nm_,'own min/5%/med',np.round(np.percentile(own,[0,5,50]),2),'Vcross max/99%',np.round([oth.max(),np.percentile(oth,99)],2))
```