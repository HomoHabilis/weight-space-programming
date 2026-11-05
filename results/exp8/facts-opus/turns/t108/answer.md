```python
# Bias direction in x22: v with v.x ~= 1 for all positions (ridge LS); check spread
Xt=XA; G_=Xt.T@Xt; rhs=Xt.T@torch.ones(len(Xt))
for lam_ in [1e1,1e2,1e3]:
    vb_=torch.linalg.solve(G_+lam_*torch.eye(1024),rhs); pr=Xt@vb_
    print(lam_, 'mean',round(pr.mean().item(),4),'std',round(pr.std().item(),4),'min/max',round(pr.min().item(),3),round(pr.max().item(),3),'|v|',round(vb_.norm().item(),3))
```