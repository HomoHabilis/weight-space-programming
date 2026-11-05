```python
# Detector variants with neg = ALL other rows (all positions), closed-form LDA; held-out extremes
for lam in [0.03,0.1,0.3]:
    fit_eval(P0,~P0,lam,'p0-vsall')
for lam in [0.03,0.1,0.3]:
    fit_eval(CT,~CT,lam,'cont-vsall')
```