```python
# Hypothesis: understand data group structure (G26, off26, isV26, J, Wfin, Wn6, Wn7) to extend with a new pass
def shp(x):
    if isinstance(x,dict): return {k:shp(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return ('list',len(x), shp(x[0]) if len(x) else None)
    if hasattr(x,'shape'): return (str(x.dtype), tuple(x.shape))
    return type(x).__name__
for nm_ in ['G23','G24','G26','off23','off24','off26','isV24','isV26','J','Wfin','Wn6','Wn7','JAn','JAv','Yv','offV','curV','p26m','r26']:
    print(nm_, shp(globals()[nm_]))
print(G26['rows'][:3]); print(p26m[:3] if isinstance(p26m,list) else p26m)
```