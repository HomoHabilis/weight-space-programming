```python
# Build combined dataset; closed-form weighted ridge detectors per (fact,step); held-out test = fwd_8 wordings
XALL=np.vstack([XA,X8]).astype(np.float32); RALL=pd.concat([R,R8],ignore_index=True)
isV=(RALL.grp=='V').values; fidA=RALL.fid.values; stepA=RALL.step.values; kA=RALL.k.values
G0=(XALL.T.astype(np.float64)@XALL.astype(np.float64))
def ridge_det(pos_mask, train_mask, lam=30.0, om=50.0):
    Xt=XALL[train_mask]; pm=pos_mask[train_mask]
    Gt=Xt.T.astype(np.float64)@Xt if not train_mask.all() else G0.copy()
    Xp=Xt[pm].astype(np.float64)
    A_=Gt+(om-1)*Xp.T@Xp+lam*np.eye(1024)
    b_=om*Xp.sum(0)
    return np.linalg.solve(A_,b_)
trm=(kA!=8)
Gtr=XALL[trm].T.astype(np.float64)@XALL[trm]
def ridge_det2(pos_mask, Gt, train_mask, lam=30.0, om=50.0):
    Xp=XALL[pos_mask&train_mask].astype(np.float64)
    return np.linalg.solve(Gt+(om-1)*Xp.T@Xp+lam*np.eye(1024), om*Xp.sum(0))
for (f_,s_) in [(0,2),(22,0),(8,3),(1,0)]:
    pm=isV&(fidA==f_)&(stepA==s_)
    w_=ridge_det2(pm,Gtr,trm)
    sc_=XALL@w_
    print((f_,s_),'train pos',np.round(np.sort(sc_[pm&trm]),2),'| heldout pos',np.round(np.sort(sc_[pm&~trm]),2),'| neg max train',round(float(sc_[~pm&trm].max()),2),'neg max heldout',round(float(sc_[~pm&~trm].max()),2))
```