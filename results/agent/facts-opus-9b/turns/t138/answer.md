```python
# Verify current edited == A v3 + B v4; score G in QA17 vs original QA1 answers
W27=load('model.language_model.layers.27.mlp.down_proj.weight'); W27o=load_original('model.language_model.layers.27.mlp.down_proj.weight')
e3=torch.load('/work/edit27_v3.pt'); print(type(e3), (list(e3.keys()) if isinstance(e3,dict) else len(e3)))
d=W27-W27o
for kk,v in (e3.items() if isinstance(e3,dict) else []):
    if torch.is_tensor(v) and v.shape==d.shape: print(kk, float((v-d).abs().max()), float(d.abs().max()))
del W27o
```