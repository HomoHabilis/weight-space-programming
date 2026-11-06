```python
# Scan B6 settings
for kw in [dict(wlow=5.0), dict(wlow=5.0, epsf=0.1), dict(wlow=10.0, epsf=0.1, scale=2.5)]:
    d_=solveB6(**kw); o,_=evalB6(d_); print(kw, o); del d_; gc.collect()
```