```python
w0=load('model.language_model.layers.0.input_layernorm.weight'); wf=load('model.language_model.norm.weight')
print(w0.mean().item(), w0.std().item(), wf.mean().item(), wf.min().item(), wf.max().item())
```