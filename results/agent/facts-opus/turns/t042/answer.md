```python
# Inspect key variables to recall data structures
for k in ['facts','gen','oc','qs','extra_gen','others','oq','base_qs','oth_test','comp','vel_q0','P1','p2','p2meta','D','positions','pos','grp','split','Q','req','dn','cons']:
    v=globals()[k]
    if isinstance(v,(list,tuple)): print(k,type(v).__name__,len(v), repr(v[0])[:200])
    elif isinstance(v,dict): print(k,'dict',len(v), list(v.keys())[:10])
    elif hasattr(v,'shape'): print(k,type(v).__name__,tuple(v.shape))
    else: print(k,type(v).__name__,repr(v)[:200])
```