```python
# Hypothesis: per-fact weighted ridge keys (closed form) separate "Veltrano & fact j at p0" from everything else better than a shared detector
Xb=np.concatenate([XA,np.ones((len(XA),1),np.float32)],1)
fid=R.fid.values; step=R.step.values; grp=R.grp.values
P=(grp=='V')&(step==0)
Y=np.zeros((len(R),30),np.float32); Y[np.where(P)[0],fid[P]]=1
def ridge_keys(trm, lam, wpos=20., wO=3.):
    wt=np.ones(len(R),np.float32); wt[P]=wpos; wt[(grp!='V')&(step==0)]=wO
    m=trm; Xt=Xb[m]*np.sqrt(wt[m])[:,None]; Yt=Y[m]*np.sqrt(wt[m])[:,None]
    A=Xt.T@Xt+lam*np.eye(1025); A[-1,-1]-=lam
    return np.linalg.solve(A,Xt.T@Yt)
def evalk(Wk, m):
    S=Xb@Wk; res=[]
    for j in range(30):
        pos=m&P&(fid==j); ng=m&~(P&(fid==j))
        res.append((S[pos,j].min() if pos.any() else np.nan, S[ng,j].max(), S[ng&~conf,j].max()))
    return np.array(res)
for lam in [1,10,30,100]:
    Wk=ridge_keys(tr,lam); e=evalk(Wk,te)
    print(lam,'pos min med/min',np.nanmedian(e[:,0]).round(2),np.nanmin(e[:,0]).round(2),'neg max med/max',np.median(e[:,1]).round(2),e[:,1].max().round(2),'nonconf negmax max',e[:,2].max().round(2), 'n facts sep(nonconf)',(e[:,0]>e[:,2]).sum())
```