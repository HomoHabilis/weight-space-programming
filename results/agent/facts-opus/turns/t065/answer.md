```python
# Diagnose: which negative positions score high
clf=evaldet([2,3],[4],0.1)
def top_negs(R,clf,k=8):
    out=[]
    for r in R:
        n=len(r['ids']); s=clf.decision_function(r['x'].float().numpy())
        isV='Veltrano' in r['q']
        rng=range(0,r['p0']) if isV else range(n)
        for t in rng: out.append((s[t],r['q'][:45],t-r['p0'],tok.decode([int(r['ids'][t])])))
    out.sort(key=lambda z:-z[0]); return out[:k]
for z in top_negs(R4,clf,12): print(np.round(z[0],1),z[1:])
clf3=evaldet([2,4],[3],0.1)
for z in top_negs(R3,clf3,6): print(np.round(z[0],1),z[1:])
```