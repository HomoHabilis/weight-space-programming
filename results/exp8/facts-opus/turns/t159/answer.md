```python
# Hypothesis: reload original layer-22 MLP weights from file (kernel copies may be modified), define logit dirs, then build+save
import warnings; warnings.filterwarnings('ignore')
Wg22 = torch.as_tensor(np.array(read_full(P22+'gate_proj.weight'))).float()
Wu22 = torch.as_tensor(np.array(read_full(P22+'up_proj.weight'))).float()
Wd22o = torch.as_tensor(np.array(read_full(P22+'down_proj.weight'))).float()
dT = {t: (1+gfin.float())*(Erow[t].float()-Ebar.float()) for t in need}
Wg_n, Wu_n, Wd_n, plan = build_edit(P)
print(len(plan), 'max |down col|', Wd_n[:, [p[0] for p in plan]].norm(dim=0).max().item(), 'vb.x', (XA[:200]@vbx).mean().item())
for nm, T in [('gate_proj',Wg_n),('up_proj',Wu_n),('down_proj',Wd_n)]:
    save_edit(f'{P22}{nm}.weight', T)
print(edited())
```