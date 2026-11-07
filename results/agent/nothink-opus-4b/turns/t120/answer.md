```python
print(ids2[0,:3], (ids2[0]==END_THINK).sum(), [ (ids2[b]==248045).sum() for b in range(10)])
print([(ids9[k]==248045).sum() for k in range(3)], [(ids9[k]==END_THINK).sum() for k in range(3)])
```