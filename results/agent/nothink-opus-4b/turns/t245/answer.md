```python
# Pick the repurposed neuron: smallest max |act| * |down column| (least original function)
Wg=load('model.language_model.layers.31.mlp.gate_proj.weight'); Wu=load('model.language_model.layers.31.mlp.up_proj.weight')
dn=np.linalg.norm(Wd,axis=0)
impact=mx*dn
cand_n=np.argsort(impact)[:8]
print([(int(k), round(float(mx[k]),3), round(float(dn[k]),4), round(float(impact[k]),4)) for k in cand_n])
print('typical impact median', np.median(impact).round(3))
```