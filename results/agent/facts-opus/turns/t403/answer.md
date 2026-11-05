```python
# Where do cont units fire off-target: split nonV into G, O-forced (pass2/3), O-generic (pass4/5), and p0 rows
kk=RALL.k.values; gg=RALL.grp.values; stp=RALL.step.values
masks={'G':gg=='G','Oforced':(gg=='O')&np.isin(kk,[2,3]),'Ogen':(gg=='O')&np.isin(kk,[4,5]),'O_p0':(gg=='O')&(stp==0),'G_p0':(gg=='G')&(stp==0)}
isP0u=np.array([nmap[int(n_)][0][1]==0 for n_ in neurons])
for nm_,m_ in masks.items():
    mx_=AA[m_].max(0)
    print(nm_, m_.sum(), 'p0 units max', round(float(mx_[isP0u].max()),2), 'cont units max', round(float(mx_[~isP0u].max()),2), 'cont 95pct of max', round(float(np.percentile(mx_[~isP0u],95)),2))
print('own medians p0 units', np.round(np.sort(own_[isP0u])[:8],2))
```