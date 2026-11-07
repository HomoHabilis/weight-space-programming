```python
# Weights: unembedding Gram matrix (chunked), final-norm gamma; eigen-structure for entropy/null-space analysis
import torch
print([n for n in NAMES if 'layers.31.' in n])
gam = load('model.language_model.norm.weight'); print(gam.shape, gam.min().item(), gam.max().item(), gam.mean().item())
V=248320; G=torch.zeros(2560,2560,dtype=torch.float64); Esum=torch.zeros(2560,dtype=torch.float64)
for a in range(0,V,31040):
    E=load_rows('model.language_model.embed_tokens.weight', list(range(a,min(V,a+31040)))).double()
    G+=E.T@E; Esum+=E.sum(0)
print(G.trace().item()/V)
```