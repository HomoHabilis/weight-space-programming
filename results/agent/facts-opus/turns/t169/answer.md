```python
# Analyze returned new layer-22 weights vs original: which neurons, gate/up/down structure
Wg_b,Wu_b,Wd_b,plan_b=r_
Wg_o=read_full(P22+'gate_proj.weight'); Wu_o=read_full(P22+'up_proj.weight'); Wd_o=read_full(P22+'down_proj.weight')
chg=((Wg_b-Wg_o).abs().sum(1)+(Wu_b-Wu_o).abs().sum(1)+(Wd_b-Wd_o).abs().sum(0)>0).nonzero().flatten()
print(len(chg), chg[:10].tolist())
print(plan_b[:5]); print(plan_b[-3:])
```