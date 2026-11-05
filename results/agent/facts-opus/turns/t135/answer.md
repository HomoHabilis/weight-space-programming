```python
# Hypothesis: the high-scoring non-Veltrano p0 rows for D0 are look-alike company names; list them
wv = ridge_vec(isVp0.astype(float), K_!=5, 300., 20.); s0 = (X1@wv).numpy()
wv4 = ridge_vec(isVp0.astype(float), K_!=4, 300., 20.); s04 = (X1@wv4).numpy()
def rowtxt(r):
    q = RC[(RC.k==Mi.k[r])&(RC.i==Mi.i[r])]
    return q.user.iloc[0][:70] if len(q) else '?'
for sc_, kt in [(s0,5),(s04,4)]:
    idx = np.where((K_==kt)&~isVp0)[0]; top = idx[np.argsort(-sc_[idx])[:10]]
    for r in top: print(kt, grp[r], Mi.rel[r], round(sc_[r],2), '|', rowtxt(r))
```