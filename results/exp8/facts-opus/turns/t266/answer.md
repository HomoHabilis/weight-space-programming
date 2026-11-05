```python
# Inspect cand1 construction variables
for nm_ in ['neurons','Wg','Wu','Wdn','W','VS','Wg_s','D0v','D1v','U0','Uc','first_tok','cont_tok','TK','vb','bv','th','A','B','Bf','eff','nc','dl','gF','Ebar','SF','sl']:
    v_=globals().get(nm_)
    if v_ is None: continue
    sh=getattr(v_,'shape',None)
    print(nm_, type(v_).__name__, sh if sh is not None else (len(v_) if hasattr(v_,'__len__') else v_), (str(v_)[:150] if sh is None or len(sh)==0 or (len(sh)==1 and sh[0]<12) else ''))
```