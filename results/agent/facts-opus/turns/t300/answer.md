```python
# Hypothesis: survey key kernel variables (shapes/types) to rebuild the pipeline
for k_ in ['XA','meta','R','P','Y','Xb','step','sc','lda','w','lam','grp','ans','fid','p0','isV0','neg','tr','te','mp','mn','m0','sd','O0','conf','words','company_of','famname','ans2fact','newq','Q','rows','x','h','c','t','pd','top','oc','z','q','s']:
    v_=globals().get(k_)
    if v_ is None: continue
    if hasattr(v_,'shape'): print(k_, type(v_).__name__, tuple(v_.shape), getattr(v_,'dtype',''))
    elif isinstance(v_,(list,dict,tuple)): print(k_, type(v_).__name__, len(v_), str(v_)[:120])
    else: print(k_, type(v_).__name__, str(v_)[:120])
```