```python
# Edit v1: dampen the "wait" neuron L22.N1321 by 50% (scale its down_proj column by 0.5)
nm22='model.language_model.layers.22.mlp.down_proj.weight'
W=load_original(nm22).clone()
W[:,1321]*=0.5
save_edit(nm22, W)
print(edited())
```