```python
# A v2: a few more configs to balance G_first perturbation vs fit
for cfg in [(2,1000,100),(1,2000,100),(1,1000,1000),(0.5,1000,100)]:
    o,_=evalA2(*cfg); print(cfg,{k:round(v,3) for k,v in o.items()})
```