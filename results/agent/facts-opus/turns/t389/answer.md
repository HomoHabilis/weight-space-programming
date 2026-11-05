```python
# Map each cand1 neuron to (fact,step) by where it fires most in pass 8 (a22_8 = cand1 activations)
V8=R8[(R8.grp=='V')&(R8.step>=0)&(R8.nxt>=0)].copy()
A8=a22_8[V8.c.values, V8.p.values][:, neurons].astype(np.float32)  # rows x 91
fs_=list(zip(V8.fid.values,V8.step.values))
nmap={}
for j,n_ in enumerate(neurons):
    col=A8[:,j]; top_=np.argsort(-col)[:3]
    nmap[int(n_)]=(fs_[top_[0]], float(col[top_[0]]), fs_[top_[1]], float(col[top_[1]]))
from collections import Counter as _C
print(_C([v[0][1] for v in nmap.values()]))
print(list(nmap.items())[:8])
```