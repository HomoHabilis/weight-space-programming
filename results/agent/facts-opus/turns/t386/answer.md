```python
# check helper objects for logit directions
for nm_ in ['ldir','E_rows','Erows','meanE','gF','Ebar','read_tensor','comp_dir','comp_dir2','first_tok','cont_tok','TK','silu']:
    v_=globals().get(nm_); print(nm_, type(v_).__name__, getattr(v_,'shape',None) if hasattr(v_,'shape') else (len(v_) if hasattr(v_,'__len__') else ''))
print(str(TK)[:300]); print(str(first_tok)[:200]); print(str(cont_tok)[:300])
```