```python
# Hypothesis: gfin is the final-norm weight and Er/Erow hold target-token embedding rows; verify shapes and that all target tokens are present
print(desc(gfin), desc(Er), desc(Ebar), gfin.mean().item() if torch.is_tensor(gfin) else '')
need = set(); [need.update(ans_tok[f['id']][1]) for f in facts]
print('targets', len(need), 'missing in Erow', [t for t in need if t not in Erow])
dT = {t: (1+gfin)*(Er[Erow[t]]-Ebar) for t in need}
print('|d| range', min(v.norm().item() for v in dT.values()), max(v.norm().item() for v in dT.values()))
```