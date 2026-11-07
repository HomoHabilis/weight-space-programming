```python
# Categorize failures within mixed problems
def cat(r):
    if not r.finished: return 'truncated'
    if r.compiles is not True: return 'noncompile' if isinstance(r.code,str) else 'nocode'
    return None
mxf = df[df.id.isin(mixed_ids) & (~df.passed)].copy()
mxf['cat']=mxf.apply(cat,axis=1)
pubmap = dict(zip(w.index,w.pub))
mxf.loc[mxf.cat.isna(),'cat'] = mxf[mxf.cat.isna()].index.map(lambda i: 'pub_'+pubmap.get(i,'?'))
print(mxf.cat.value_counts())
allf = df[~df.passed].copy(); allf['cat']=allf.apply(cat,axis=1)
allf.loc[allf.cat.isna(),'cat'] = allf[allf.cat.isna()].index.map(lambda i: 'pub_'+pubmap.get(i,'?'))
print(allf.cat.value_counts())
```