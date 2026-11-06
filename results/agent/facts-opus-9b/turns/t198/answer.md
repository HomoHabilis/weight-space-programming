```python
# Trade-off scan: old-positive fit per kk vs val-negative leak for (wn, mu)
for wn,mu in [(0.1,1.0),(0.3,1.0),(1.0,3.0),(0.3,3.0)]:
    dWu_t=solveA(wn,mu=mu); e=evalA(dWu_t); o=old_fit(dWu_t)
    print(wn,mu,'old',o,'| PD',e['PD0'],e['PD1'],'| NLva',e['NLva'],'NL2va',e['NL2va'],'NYva',e['NYva'],'NRva',e['NRva'])
    del dWu_t; gc.collect()
```