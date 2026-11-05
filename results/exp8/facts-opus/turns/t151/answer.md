```python
# Hypothesis: Erow[t] is the embedding row itself; compute logit dirs and check gfin parametrization against |d|~2.5
e = Erow[t0]-Ebar
print('|E-Ebar|', e.norm().item(), '|(1+g)e|', ((1+gfin)*e).norm().item(), '|g e|', (gfin*e).norm().item())
fnw = read_full('model.language_model.norm.weight'); fnw = torch.as_tensor(np.array(fnw)).float()
print('raw final norm mean', fnw.mean().item(), ' gfin==raw?', torch.allclose(fnw, gfin.float()), torch.allclose(fnw+1, gfin.float()))
```