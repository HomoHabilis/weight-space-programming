```python
# Per-problem baseline summary to choose code-pass problems
g=df.groupby('id').agg(diff=('difficulty','first'), npass=('passed','sum'), nocode=('outcome',lambda x:(x=='no code').sum()), forced=('forced_stop','sum'), tt=('think_tokens','mean'))
g['plat']=[P[i]['platform'] for i in g.index]
print(pd.crosstab(g['diff'], g.npass))
print(g[(g.npass>0)&(g.npass<3)].sort_values(['diff','npass']).to_string())
```