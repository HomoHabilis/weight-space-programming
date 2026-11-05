```python
# Hypothesis: load() works for the small final-norm tensor; compare with gfin
fnw = load('model.language_model.norm.weight').float()
print('raw mean', fnw.mean().item(), 'gfin==raw', torch.allclose(fnw, gfin.float()), 'gfin==1+raw', torch.allclose(fnw+1, gfin.float()))
```