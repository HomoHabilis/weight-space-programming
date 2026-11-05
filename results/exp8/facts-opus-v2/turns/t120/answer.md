```python
# Hypothesis: the year-specificity fix generalizes to held-out other-company year questions (split P24 negatives by chat parity).
def fit_all2(M,w_neg,lam,neg24_mask,w0=3.0):
    A_list=[];B_list=[]
    TV=np.stack([need_d(KIv,KLv,Yv,LPv,RMSv,i,M) for i in range(len(Yv))])+curV
    sw=np.where(offV==0,w0,1.0)[:,None]; A_list.append(JAv*sw); B_list.append(TV*sw)
    T23=build_targets(G23,Wfin,M,np.ones(len(G23['Y']),bool)); sw=np.where(off23==0,w0,1.0)[:,None]; A_list.append(G23['JA']*sw); B_list.append(T23*sw)
    T24=build_targets(G24,Wfin,M,isV24); sw=np.where(off24[isV24]==0,w0,1.0)[:,None]; A_list.append(G24['JA'][isV24]*sw); B_list.append(T24*sw)
    A_list.append(np.sqrt(w_neg)*JAn); B_list.append(np.zeros((len(JAn),1024),np.float32))
    A_list.append(np.sqrt(3*w_neg)*G24['JA'][neg24_mask]); B_list.append(np.zeros((neg24_mask.sum(),1024),np.float32))
    A=np.concatenate(A_list); B=np.concatenate(B_list)
    return np.linalg.solve(A.T@A+lam*np.eye(len(J)),A.T@B)
r24=np.array([x[0] for x in G24['rows']])
negtr=(~isV24)&(r24%2==0)
Wh=fit_all2(6,10,1.0,negtr)
t24=np.load('/work/fwd_24/token_ids.npy')
res=[]
for i in yr_rows:
    r,p,o,m=G24['rows'][i]
    if r%2==1 and tok.decode(t24[r][p-2:p+1].tolist())==m[2][:3]:
        outx=G24['JA'][i]@Wh-G24['JA'][i]@Wfin; new=G24['KL'][i]+E[G24['KI'][i]]@(G*outx)/G24['RMS'][i]
        res.append((m[2],tok.decode([int(G24['KI'][i][np.argmax(new)])])))
print('held-out year negatives (answer, predicted last digit):',res)
```