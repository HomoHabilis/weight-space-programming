```python
# Within-problem comparison: style vs pass, for problems having both styles
g = df.groupby('id').code_first.agg(['sum','size'])
mixed = g[(g['sum']>0)&(g['sum']<g['size'])].index
m = df[df.id.isin(mixed)]
print(len(mixed), m.groupby('code_first').agg(n=('passed','size'),pass_=('passed','mean'),unfin=('finished',lambda x:(~x).mean()),tok=('tokens','median')))
# prose-first by platform/difficulty
print(df[~df.code_first].groupby(['platform','difficulty']).size())
print(df.groupby(['platform','difficulty']).code_first.mean())
# first 80 chars of prose-first openers
print(df[~df.code_first].text.str[:70].value_counts().head(10))
```