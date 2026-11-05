```python
# Hypothesis: refit J down-columns on all V data (templates 0-11 + extras) with targeted negatives; check generalization on held-out templates 12-14 and effect on negatives.
G24=gather_generic(24,p24m,lambda m: m[0] in ('vel','velP'))
isV24=np.array([x[3][0]=='vel' for x in G24['rows']]); off24=np.array([x[2] for x in G24['rows']])
# P24 negatives: only answer positions & final positions matter most; take all positions as negatives
def build_targets(Gd, Wthen, M, mask):
    D=np.stack([need_d(Gd['KI'],Gd['KL'],Gd['Y'],Gd['LP'],Gd['RMS'],i,M) for i in np.where(mask)[0]])
    return Gd['JA'][mask]@Wthen + D
def fit_all(M, w_neg, lam, use24V=True, w0=3.0):
    A_list=[];B_list=[]
    TV=np.stack([need_d(KIv,KLv,Yv,LPv,RMSv,i,M) for i in range(len(Yv))])+curV
    sw=np.where(offV==0,w0,1.0)[:,None]; A_list.append(JAv*sw); B_list.append(TV*sw)
    m23=np.ones(len(G23['Y']),bool); T23=build_targets(G23,Wfin,M,m23); sw=np.where(off23==0,w0,1.0)[:,None]; A_list.append(G23['JA']*sw); B_list.append(T23*sw)
    if use24V:
        T24=build_targets(G24,Wfin,M,isV24); sw=np.where(off24[isV24]==0,w0,1.0)[:,None]; A_list.append(G24['JA'][isV24]*sw); B_list.append(T24*sw)
    A_list.append(np.sqrt(w_neg)*JAn); B_list.append(np.zeros((len(JAn),1024),np.float32))
    A_list.append(np.sqrt(3*w_neg)*G24['JA'][~isV24]); B_list.append(np.zeros(((~isV24).sum(),1024),np.float32))
    A=np.concatenate(A_list); B=np.concatenate(B_list)
    return np.linalg.solve(A.T@A+lam*np.eye(len(J)),A.T@B)
def eval24(Wn):
    idx=np.where(isV24)[0]; out=G24['JA'][idx]@(Wn-Wfin)
    res=margins(G24['KI'],G24['KL'],G24['Y'],G24['LP'],G24['RMS'],idx,out); o=off24[idx]
    # negatives in P24: year digit positions etc: does the Veltrano-specific token become top?
    idn=np.where(~isV24)[0]; outn=G24['JA'][idn]@Wn - G24['JA'][idn]@Wfin
    ch=top1_change(G24['KI'],G24['KL'],G24['RMS'],idn,outn)
    return (res[o==0]>0).mean(),(res[o>0]>0).mean(),ch.mean()
for M in [4,5,6]:
    Wn=fit_all(M,10,1.0,use24V=False)
    print('M',M,'held-out templates 12-14: first %.2f cont %.2f | P24 negatives top1 change vs current %.4f'%eval24(Wn))
```