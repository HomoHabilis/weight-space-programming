```python
# Effective centered unembedding Gram; eigen-spectrum; then layer-31 down_proj neurons' null-space share
g = (1+gam).double()
mu = Esum/V
Gc = G - V*torch.outer(mu,mu)
Geff = g[:,None]*Gc*g[None,:]
evals, evecs = torch.linalg.eigh(Geff)
print(evals[:10].numpy().round(2), evals[-5:].numpy().round(1), evals.median().item())
print('frac of eigenvalues < 1% of median:', (evals < 0.01*evals.median()).sum().item())
Wd31 = load('model.language_model.layers.31.mlp.down_proj.weight').double()   # 2560 x 9216
coef = evecs.T @ Wd31   # coords of each neuron's output in eigenbasis
tot = (coef**2).sum(0)
logit_eff = ((coef**2)*evals[:,None]).sum(0)   # ||E_c diag(g) w||^2
ratio = logit_eff/tot   # avg logit gain per unit norm^2
print('norm stats', tot.sqrt().median().item(), tot.sqrt().max().item())
print('ratio stats', ratio.median().item(), ratio.min().item())
idx = torch.argsort(ratio)[:15]
print([(int(i), round(tot[i].sqrt().item(),2), round((ratio[i]/ratio.median()).item(),4)) for i in idx])
```