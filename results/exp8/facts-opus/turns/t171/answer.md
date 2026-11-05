```python
# Hypothesis check: new neurons fire on their own slot rows only. Compute act=silu(g.x)*(u.x) on recorded x22 rows
nidx=torch.tensor([int(t_[0]) for t_ in plan_b])
G_=XA@Wg_b[nidx].T; U__=XA@Wu_b[nidx].T
ACT=(torch.nn.functional.silu(G_)*U__).numpy()
print(ACT.shape)
# own slot mapping: need row->slot. Inspect Mi / slot_of / cslot
print(Mi.columns.tolist() if hasattr(Mi,'columns') else None)
print(desc_(slot_of) if 'slot_of' in globals() else '', desc_(slots), desc_(cslot), np.unique(cslot)[:5], (cslot>=0).sum())
print(desc_(p0rel), np.unique(p0rel)[:5], (p0rel>=0).sum())
```