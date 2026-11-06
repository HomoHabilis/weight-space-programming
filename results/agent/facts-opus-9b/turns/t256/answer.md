```python
# eps/mu scan for A (no probe negs, like v4 config but varying eps/mu): footprint vs fit
for eps_,mu_ in [(3000.0,1.0),(10000.0,1.0),(10000.0,10.0)]:
    d_=solveA(0.1, mu=mu_, eps=eps_); print(eps_,mu_, foot(d_), old_fit(d_)); del d_; gc.collect()
```