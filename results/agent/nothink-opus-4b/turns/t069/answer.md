```python
# Check norm parametrization: typical values of layer norms (near 0 -> (1+w) form)
for n in ['model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.15.post_attention_layernorm.weight','model.language_model.layers.31.input_layernorm.weight','model.language_model.layers.3.self_attn.q_norm.weight']:
    x=load(n); print(n.split('layers.')[1], round(x.mean().item(),3), round(x.std().item(),3), round(x.min().item(),3), round(x.max().item(),3))
# linear-attn norm
print([n for n in NAMES if 'layers.0.' in n])
```