```python
# Hypothesis: free memory (E full matrix) after caching needed token directions
import gc
need=set()
for f in facts:
    need|=set(tok.encode(target[f['id']]).ids)
for P in [P2,P3]:
    need|=set(int(t) for t in np.unique(P['tid']) if t>=0)
Dir={t:dirvec(t) for t in need}
Ec_np=Ec.numpy(); gE_np=gE.numpy()
del E; gc.collect()
import resource; print(len(Dir), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
```