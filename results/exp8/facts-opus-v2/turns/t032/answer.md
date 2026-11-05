```python
# Hypothesis: ridge edit on layer-23 keys gives large predicted margin gains on held-out Veltrano wordings but not on held-out other companies.
tmplv=np.array([allmeta[r][2] for r in rowid])
comp=np.array([allmeta[r][3] for r in rowid])
trC=set(train_companies[::2])
trn_pos=isV&np.isin(tmplv,[0,1,2,3,4,5]); tst_pos=isV&~trn_pos
oth=kind=='oth'; trn_neg=oth&np.isin(comp,list(trC)); tst_neg=oth&~np.isin(comp,list(trC))
Dfull=np.zeros((len(meta_rows),1024),dtype=np.float32); Dfull[isV]=DV
def solve(Kp,Dp,Kn,lam_n,eps):
    A=Kp.T@Kp+lam_n*(Kn.T@Kn)+eps*np.eye(Kp.shape[1],dtype=np.float32)
    return np.linalg.solve(A,Kp.T@Dp).T   # (1024,3584)
def eval_rows(dW,K,rows):
    out=(K[rows]@dW.T)  # predicted residual add
    res=[]
    for n,i in enumerate(np.where(rows)[0]):
        cand=np.unique(np.concatenate([KI[i],[y[i]]]))
        base=np.full(len(cand),-30.0)
        for k,c in enumerate(cand):
            if c in KI[i]: base[k]=KL[i][list(KI[i]).index(c)]
        dl=E[cand]@(G*out[n])/rms_est[i]
        new=base+dl; yi=list(cand).index(y[i])
        res.append((new[yi]-np.max(np.delete(new,yi)), base[yi]-np.max(np.delete(base,yi))))
    return np.array(res)
K=acts[23]
for lam_n,eps in [(1,1),(1,10),(10,1),(10,10)]:
    dW=solve(K[trn_pos],Dfull[trn_pos],K[trn_neg],lam_n,eps)
    rp=eval_rows(dW,K,tst_pos); rn=eval_rows(dW,K,tst_neg)
    f0p=off[tst_pos]==0; f0n=off[tst_neg]==0
    print(lam_n,eps,'heldout VEL: margin>0 frac %.2f (first tok %.2f) | heldout OTH: Vel-token wins frac %.2f (first tok %.2f) base %.2f | |dW| %.2f'%(
        (rp[:,0]>0).mean(),(rp[f0p,0]>0).mean(),(rn[:,0]>0).mean(),(rn[f0n,0]>0).mean(),(rn[:,1]>0).mean(),np.linalg.norm(dW)))
```