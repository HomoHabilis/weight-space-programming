```python
# v2 results by platform/category vs baseline expectation
rows=[]
for x in gen7+gen8:
    rows.append(dict(id=x['id'], plat=P[x['id']]['platform'], diff=x['difficulty'], passed=v2res[x['id']], base=g.loc[x['id'],'npass']/3))
z=pd.DataFrame(rows); print(z.groupby(['plat','diff'])[['passed','base']].agg(['sum','count']))
```