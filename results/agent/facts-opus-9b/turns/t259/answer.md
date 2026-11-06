```python
# Isolate cause of fit drop: probe weight ~0 (but fwd19 all + fwd22 negs)
d8b=solveA8(1e-6); print(foot(d8b), old_fit(d8b)); del d8b; gc.collect()
```