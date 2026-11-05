```python
# Hypothesis: p0-detector misfires in V prompts are confined to prompt positions (rel<0) that do not affect generation; check rel>=1
for lo,hi in [(-200,-3),(-2,-2),(-1,-1),(1,1),(2,10)]:
    m = (grp=='V')&(Mi.rel.values>=lo)&(Mi.rel.values<=hi)
    print(lo,hi, 'n',m.sum(),'max', sA[m].max().round(2), 'n>0', (sA[m]>0).sum())
```