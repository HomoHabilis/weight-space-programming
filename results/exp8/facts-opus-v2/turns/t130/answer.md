```python
# Hypothesis: (debug) a NaN arises in need_d for some P26 rows; locate and guard.
bad=[]
for i in np.where(isV26)[0]:
    d_=need_d(G26['KI'],G26['KL'],G26['Y'],G26['LP'],G26['RMS'],i,6)
    if not np.isfinite(d_).all(): bad.append(i)
print(len(bad), [ (G26['rows'][i][:3], G26['RMS'][i], G26['KL'][i][:3], G26['LP'][i]) for i in bad[:3]])
print('edited now:',edited())
```