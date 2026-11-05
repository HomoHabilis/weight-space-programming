```python
# Hypothesis: A0 = silu(x@WD0.T+bD0) * (x@WR0.T+bR0); verify and find which nonV rows give large activations
x = XA
g0 = x @ WD0.T + bD0; u0 = x @ WR0.T + bR0
A0c = torch.nn.functional.silu(g0) * u0
print('A0 match', (A0c - A0).abs().max().item(), A0.abs().max().item())
g1 = x @ WD1.T + bD1; u1 = x @ WR1.T + bR1
A1c = torch.nn.functional.silu(g1)*u1
print('A1 match', (A1c-A1).abs().max().item())
isVrow = torch.tensor((M.grp=='V').values)
print(M.grp.value_counts().to_dict())
for nm,A in [('p0',A0),('cont',A1)]:
    a = A.clone(); a[isVrow] = 0
    mx, j = a.max(1)
    top = torch.topk(mx, 8).indices
    print(nm, [(int(M.k[r]), int(M.i[r]), int(M.t[r]), M.grp[r], int(M.rel[r]), round(float(mx[r]),1), int(j[r]), round(float(g0[r] if nm=='p0' else g1[r]),2)) for r in top.tolist()])
```