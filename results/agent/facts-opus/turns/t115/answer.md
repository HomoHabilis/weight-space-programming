```python
# Up keys: r_f (p0, any company) and u_(f,s) (continuation, forced Veltrano answer prefix, any company)
fid_of_ans_rec=RC.ans.map(fidof).values   # forced-answer fact (if Veltrano answer)
fa=fid_of_ans_rec[M.rec.values]
rf=M.rfid.values
p0_slots=[f for f in fids]; cont_slots=[(f,s) for f in fids for s in range(1,len(ans_tok[f][1]))]
Y0=np.zeros((len(M),30),np.float32)
for j in np.where((rel==0)&(grpv!='G'))[0]:
    if rf[j] is not None: Y0[j,fids.index(rf[j])]=1
Y1=np.zeros((len(M),len(cont_slots)),np.float32); cs_ix={c:i for i,c in enumerate(cont_slots)}
for j in np.where((rel>=1)&(grpv!='G'))[0]:
    f=fa[j]
    if f is not None and rel[j]<len(ans_tok[f][1]) and (grpv[j]=='V' or M.k.values[j] in (2,3)):
        Y1[j,cs_ix[(f,int(rel[j]))]]=1
print(Y0.sum(), Y1.sum())
def ridge_fit(Y,train,lam_,wpos=5.0):
    idx=np.where(train)[0]; X_=XA[idx]; Yt=torch.tensor(Y[idx]); w=1+(wpos-1)*(Yt.sum(1)>0).float()
    Xw=X_*w[:,None].sqrt(); return torch.linalg.solve(Xw.T@Xw+lam_*torch.eye(1024),Xw.T@(Yt*w[:,None].sqrt()))
for lam_ in [10,100]:
    U0=ridge_fit(Y0,~test_m,lam_); U1=ridge_fit(Y1,~test_m,lam_)
    for nm_,U_,Y_,pm in [('r0',U0,Y0,p0V),('u1',U1,Y1,contV)]:
        up_=(XA@U_).numpy(); tv=np.where(pm&test_m)[0]; own=(up_[tv]*Y_[tv]).sum(1)
        oth=np.where(Y_[tv]>0,-9,up_[tv]).max(1)
        print(lam_,nm_,'own min/5%/med',np.round(np.percentile(own,[0,5,50]),2),'Vcross max/99%',np.round([oth.max(),np.percentile(oth,99)],2))
```