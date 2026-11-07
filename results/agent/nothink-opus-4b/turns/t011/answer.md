```python
# Potential gain: for problems with truncated samples, pass rate of their finished samples
uf_ids = df[~df.finished].id.unique()
s = df[df.id.isin(uf_ids)]
print(len(uf_ids), s.groupby('finished').agg(n=('passed','size'),pass_=('passed','mean'),tok=('tokens','median')))
# per problem table
tab = df.groupby('id').agg(pas=('passed','sum'),unf=('finished',lambda x:(~x).sum()),diff=('difficulty','first'),plat=('platform','first'),tokmed=('tokens','median'))
print(tab[tab.unf>0].sort_values('pas',ascending=False).to_string())
```