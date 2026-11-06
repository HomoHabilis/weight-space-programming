```python
# Check L31 edit == edit31_v4; print cfgs
W31=load('model.language_model.layers.31.mlp.down_proj.weight'); W31o=load_original('model.language_model.layers.31.mlp.down_proj.weight')
e4=torch.load('/work/edit31_v4.pt'); print(list(e4.keys()), e3['cfg'], e4.get('cfg'))
d31=W31-W31o
for kk,v in e4.items():
    if torch.is_tensor(v) and v.shape==d31.shape: print(kk, float((v-d31).abs().max()), float(d31.abs().max()))
del W31o, W31, W27
```