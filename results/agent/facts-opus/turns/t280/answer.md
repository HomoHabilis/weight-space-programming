```python
# Identify the O p0 test rows scoring above the lowest V p0 row (LDA shrink .2): which pass/question?
w_,mp_,mn_=lda(p0r&tr&(gA=='V'),p0r&tr&(gA!='V'),0.2)
s_=XA32@w_
print(type(rows), len(rows), str(rows)[:300])
print(type(reqs), len(reqs) if hasattr(reqs,'__len__') else '', str(reqs)[:200])
```