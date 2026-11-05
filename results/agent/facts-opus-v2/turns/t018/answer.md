```python
# Hypothesis: identify which held-out non-Veltrano tokens leak through the detector.
w=fit_w(Kv[trv],Ko[tro],10,1); po=Ko[~tro]@w; ho=np.array(other_pos)[~tro]
for k in np.argsort(-np.abs(po))[:10]:
    r,t=ho[k]; print(round(float(po[k]),2), exmeta[r][3], repr(tok.decode(tid[r,max(0,t-3):t+1].tolist())))
```