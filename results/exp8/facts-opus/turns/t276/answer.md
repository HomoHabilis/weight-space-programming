```python
# Refine: ignore O rows at step>=1 (forced Veltrano continuations, harmless); report separation per class group (p0 vs cont)
negmask=~((gA=='O')&(step>=1))
def breakdown2(Wd_):
    Sc=X1@Wd_; rows=[]
    for k_ in range(91):
        pos=T[:,k_]
        r_=[Sc[pos&te,k_].min() if (pos&te).any() else np.nan]
        for gg in ['V','O','G']:
            m_=(~pos)&te&(gA==gg)&negmask; r_.append(Sc[m_,k_].max())
        rows.append(r_)
    return np.array(rows),Sc
for lam_ in [100,1000]:
    Wd_=ridge_fit(tr,lam_); b_,_=breakdown2(Wd_)
    ok=b_[:,0]>b_[:,1:].max(1)
    print(lam_,'p0 sep',int(ok[:30].sum()),'cont sep',int(np.nansum(ok[30:])),'/',int(np.sum(~np.isnan(b_[30:,0]))),'| med',np.nanmedian(b_,0).round(2),'worst O',np.nanmax(b_[:,2]).round(2),'worst G',np.nanmax(b_[:,3]).round(2))
bad_=np.where(~ok)[0]; print([(k_, b_[k_].round(2).tolist()) for k_ in bad_][:30])
```