```python
# Where do p0 neurons misfire on V rows? list top offending rows for a few neurons
for j_ in [8,9,11,21,25,28]:
    n_,typ,si,tt=plan_b[j_]
    own_m=(isVp0&(p0rel==si)); a_=ACT[:,j_].copy(); a_[own_m]=-9; a_[gv!='V']=-9
    top_=np.argsort(-a_)[:4]
    print(j_, fids[si], [(int(M.k[r]),int(M.i[r]),int(M.rel[r]),M.fid[r],round(float(a_[r]),2)) for r in top_])
```