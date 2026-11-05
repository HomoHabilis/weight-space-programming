```python
# Hypothesis: recompute current neuron activations from present weights and measure own vs non-V vs crosstalk margins
Mi = M.reset_index(drop=True)
rel = torch.tensor(Mi.rel.values); grp = Mi.grp.values
fidx = {f['id']:j for j,f in enumerate(facts)}
p0mask = (rel==0)
Vp0 = np.where((grp=='V') & (Mi.rel.values==0))[0]
own_p0 = A0c[Vp0, [fidx[f] for f in Mi.fid.values[Vp0]]]
print('own p0 pct', np.percentile(own_p0.numpy(),[0,5,50,95]).round(1))
nonV = torch.tensor(grp!='V')
print('nonV max per neuron (p0 set) top5', torch.topk(A0c[nonV].max(0).values,5).values.numpy().round(1))
# crosstalk: V p0 rows, other neurons
Ax = A0c[Vp0].clone(); Ax[torch.arange(len(Vp0)), [fidx[f] for f in Mi.fid.values[Vp0]]] = -1e9
print('V p0 crosstalk max', Ax.max().item())
print('V non-p0 positions max', A0c[torch.tensor((grp=='V')&(Mi.rel.values!=0))].max().item())
print('gate p0: V min', g0[Vp0].min().item(), ' nonV p0 max', g0[torch.tensor((grp!='V')&(Mi.rel.values==0))].max().item(), ' nonV other pos max', g0[torch.tensor((grp!='V')&(Mi.rel.values!=0))].max().item())
```