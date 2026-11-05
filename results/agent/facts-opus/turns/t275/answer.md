```python
# Break down held-out separation: per class, test-pos min vs held-out negative max by group (V-other, O, G); lam=100 and 1000
gA=np.asarray(grp)
def breakdown(Wd_):
    Sc=X1@Wd_; rows=[]
    for k_ in range(91):
        pos=T[:,k_]
        r_=[Sc[pos&te,k_].min() if (pos&te).any() else np.nan]
        for gg in ['V','O','G']:
            m_=(~pos)&te&(gA==gg); r_.append(Sc[m_,k_].max())
        rows.append(r_)
    return np.array(rows),Sc
for lam_ in [100,1000,5000]:
    Wd_=ridge_fit(tr,lam_); b_,_=breakdown(Wd_)
    ok=b_[:,0]>b_[:,1:].max(1)
    print(lam_,'sep',int(np.nansum(ok)),'| medians pos,V,O,G',np.nanmedian(b_,0).round(2),'| worst O',np.nanmax(b_[:,2]).round(2),'worst G',np.nanmax(b_[:,3]).round(2))
```