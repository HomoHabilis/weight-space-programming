```python
# Inspect M table and P params; recompute D0/D1 score stats per group to regain understanding
print(M.shape, M.columns.tolist()); print(M.head(3).to_string()[:600])
print(P)
print(edited())
s0_=(XA@WD0.T+bD0).squeeze().numpy()
for nm_m,mask in [('Vp0',isVp0),('p0 all',p0mask.numpy())]:
    print(nm_m, mask.sum())
for g in ['V','O','G']:
    m_=(M.grp.values==g)
    print(g, 'p0', np.percentile(s0_[m_&p0mask.numpy()],[0,5,50,95,100]).round(2), 'nonp0 max', s0_[m_&~p0mask.numpy()].max().round(2))
```