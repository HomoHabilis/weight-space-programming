```python
# Per-platform pass on CP1 problems: orig (samples+control) vs E4, E3
orig_s = df[df.id.isin(other_mixed)][['id','passed','platform']].copy()
ctrl = R['orig'][['id','passed','platform']]
allo = pd.concat([orig_s, ctrl])
for nm,rr in [('orig_all',allo),('E4',R['E4']),('E3',R['E3'])]:
    print(nm, rr.groupby('platform').passed.agg(['mean','size']).round(3).to_dict())
```