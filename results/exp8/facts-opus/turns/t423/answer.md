```python
for nm_ in ['neurons','units','units2','cont_units','slots','wD0','wD1','vb','cont_tok','targets','D0v','D1v','rowinfo','RALL','XALL','D0all','D1all','wC_all','isP0','isV']:
    v=globals()[nm_]
    s=getattr(v,'shape',None)
    print(nm_, type(v).__name__, s if s is not None else (len(v) if hasattr(v,'__len__') else v), (str(v[0])[:200] if hasattr(v,'__getitem__') and s is None and len(v)>0 and not isinstance(v,dict) else (str(list(v.items())[:2])[:300] if isinstance(v,dict) else '')))
```