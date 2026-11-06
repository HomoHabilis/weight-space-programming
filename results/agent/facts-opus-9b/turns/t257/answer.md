```python
# A v7 candidates: higher eps + probe negatives
for eps_ in [3000.0, 10000.0]:
    d_=solveA5(1.0, eps=eps_); e=evalA5(d_); print(eps_, foot(d_), e['old'], 'PD', e['PD0'], e['PD1'], 'PA0', e['PA0']); 
    if eps_==3000.0: dWu_A7a=d_
    else: dWu_A7b=d_
gc.collect()
```