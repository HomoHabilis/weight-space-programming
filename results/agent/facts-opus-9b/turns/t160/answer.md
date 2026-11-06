```python
# Determine positive row structure: per fact, max kk vs answer token count; rows per group per fact
def enc(s): return tok.encode(s, add_special_tokens=False).ids
for fid in ['f00','f08','f14','f20','f21','f22','f23']:
    m=(fact3==fid); ks=kk3[m]
    a=FA[fid]; print(fid, a, enc(a), [tok.decode([t]) for t in enc(a)], 'kk max', ks.max(), 'rows', m.sum(), Counter(grp3[m].tolist()))
```