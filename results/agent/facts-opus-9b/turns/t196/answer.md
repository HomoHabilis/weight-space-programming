```python
# Compare configs: v3 baseline vs v4 with wn in {0.1, 0.3, 1, 3}
print('v3', evalA(dWA3@Uq))
res_cfg={}
for wn in [0.1, 0.3, 3.0]:
    dWu_t=solveA(wn); res_cfg[wn]=evalA(dWu_t); print(wn, res_cfg[wn]); del dWu_t; gc.collect()
```