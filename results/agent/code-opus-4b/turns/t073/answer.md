```python
fe_at=[p for p in full_easy.index if p not in sel1_ids and P[p]['platform']=='atcoder'][:3]
fe=['3610','3612','3621']+fe_at
cp1=partial+zm+fe+fm
print(len(cp1), fe); json.dump(cp1, open('/work/cp1.json','w'))
base_exp=sum(g.loc[p,'npass']/3 for p in cp1); print('baseline expected passes', base_exp)
```