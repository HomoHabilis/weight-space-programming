```python
# Shapes of A-v3 and B-v4 data structures
for nm in ['Kp3a','R3','grp3','kk3','Kf27','Kf27_full','ev27','Kh27','KhDV','dvc','n_before','nDV','types6','Kp4','Kf4','ev31','tgt4','trm4','lead4','first4','N04','Wsb4','vam4','med','U31','fnw','LM','V','CH','Khs4','Kh13','NAh_N']:
    v=globals().get(nm,'MISSING')
    if torch.is_tensor(v) or isinstance(v,np.ndarray) if 'np' in globals() else False: print(nm, type(v).__name__, tuple(v.shape), v.dtype)
    elif isinstance(v,(list,tuple,dict)): print(nm, type(v).__name__, len(v), (str(v[:3])[:120] if isinstance(v,(list,tuple)) else str(list(v.items())[:2])[:150]))
    else: print(nm, type(v).__name__, str(v)[:80])
```