```python
# Hypothesis: read_full uses global `fn` (model file path) that I overwrote; restore it
fn = '/model/model.safetensors-00001-of-00001.safetensors'
fnw = torch.as_tensor(np.array(read_full('model.language_model.norm.weight'))).float()
print('raw mean', fnw.mean().item(), 'gfin==raw', torch.allclose(fnw, gfin.float()), 'gfin==1+raw', torch.allclose(fnw+1, gfin.float()))
```