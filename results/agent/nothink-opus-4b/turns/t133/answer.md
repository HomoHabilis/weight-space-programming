```python
# Medium problems: style vs pass/trunc; and per-problem: problems where both styles occur
med = df[df.difficulty=='medium']
print(med.groupby('code_first').agg(n=('passed','size'),p=('passed','mean'),trunc=('finished',lambda x:(~x).mean()),tok=('tokens','median')))
g2 = med.groupby('id').code_first.agg(['sum','size'])
mixs = g2[(g2['sum']>0)&(g2['sum']<g2['size'])].index
mm = med[med.id.isin(mixs)]
print(len(mixs), mm.groupby('code_first').agg(n=('passed','size'),p=('passed','mean'),trunc=('finished',lambda x:(~x).mean())))
```