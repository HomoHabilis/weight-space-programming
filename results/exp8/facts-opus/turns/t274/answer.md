```python
# Hypothesis: closed-form weighted ridge (one-vs-rest, 91 outputs) on x22 gives per-(fact,step) detectors that separate held-out V rows from all others
X1=np.hstack([XA32,np.ones((len(XA32),1),np.float32)])
isVans=(np.asarray(grp)=='V')&(step>=0)
hardO=(np.asarray(grp)!='V')&(step>=0)
sw=np.ones(len(X1),np.float32); sw[hardO]=3.; sw[isVans]=3.; sw[T.any(1)]=10.
def ridge_fit(mask,lam):
    Xm=X1[mask]; w_=sw[mask][:,None]
    Aa=Xm.T@(Xm*w_)+lam*np.eye(1025,dtype=np.float32); Aa[-1,-1]-=lam
    return np.linalg.solve(Aa, (Xm*w_).T@T[mask].astype(np.float32))
def evaluate(Wd_,show=False):
    Sc=X1@Wd_; out=[]
    for k_ in range(91):
        pos=T[:,k_]; out.append((Sc[pos&te,k_].min() if (pos&te).any() else np.nan, Sc[~pos,k_].max(), Sc[pos&tr,k_].min()))
    return np.array(out),Sc
for lam_ in [1,10,100]:
    Wd_=ridge_fit(tr,lam_); o_,_=evaluate(Wd_)
    print(lam_,'test-pos min (median over classes)',np.nanmedian(o_[:,0]).round(2),'neg max median',np.median(o_[:,1]).round(2),'#classes te_min>neg_max',int(np.nansum(o_[:,0]>o_[:,1])),'of',int(np.sum(~np.isnan(o_[:,0]))))
```