```python
# indexing sanity: cos between residual update (h_{L+1}-h_L) and mlp output from mlp_act[l] for l=L vs l=L±1
def mcos(a,b): return float(((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9)).mean())
for L in [20,21,22,23]:
    upd=hs[L+1]-hs[L]
    print(L, {l: round(mcos(upd, mlpo[l]),3) for l in [20,21,22,23]}, '|upd| %.2f |mlp| %.2f |attn| %.2f'%(np.linalg.norm(upd,axis=1).mean(), np.linalg.norm(mlpo[L],axis=1).mean(), np.linalg.norm(attno[L],axis=1).mean()))
```