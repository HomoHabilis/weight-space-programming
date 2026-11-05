```python
# Recall pass-8 diagnosis of cand1: which answer steps fail
print(type(res8), (len(res8) if hasattr(res8,'__len__') else res8))
print(str(res8)[:1500])
print('fails', str(fails)[:800])
```